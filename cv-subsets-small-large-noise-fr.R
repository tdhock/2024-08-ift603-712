library(data.table)
n.folds <- 6
max.N <- 2000
N <- max.N*n.folds/(n.folds-1)
abs.x <- 3*pi
set.seed(2)
norm01 <- function(z,ref=z)(z-min(ref))/(max(ref)-min(ref))
(grid.dt <- data.table(
  raw=seq(-abs.x,abs.x, l=201),
  y=0 #for mlr3
)[, x := norm01(raw)][])
x.vec <- runif(N, -abs.x, abs.x)
str(x.vec)
reg.pattern.list <- list(
  sin=sin)
standard.deviation.vec <- c(
  easy=0.1,
  hard=1.7)
reg.task.list <- list()
reg.data.list <- list()
grid.signal.dt.list <- list()
for(signal in names(reg.pattern.list)){
  f <- reg.pattern.list[[signal]]
  for(difficulty in names(standard.deviation.vec)){
    standard.deviation <- standard.deviation.vec[[difficulty]]
    task_id <- paste(signal, difficulty)
    signal.vec <- f(x.vec)
    y <- signal.vec+rnorm(N,sd=standard.deviation)
    task.dt <- data.table(
      x=norm01(x.vec,grid.dt$raw), y)
    reg.data.list[[paste(difficulty, task_id)]] <- data.table(
      difficulty,
      signal,
      task_id,
      task.dt)
    reg.task.list[[paste(difficulty, task_id)]] <- mlr3::TaskRegr$new(
      task_id, task.dt, target="y"
    )
    grid.signal.dt.list[[paste(difficulty, task_id)]] <- data.table(
      difficulty,
      signal,
      task_id,
      algorithm="ideal",
      x=grid.dt$x,
      y=f(grid.dt$raw)
    )
  }
}
(reg.data <- rbindlist(reg.data.list))
(grid.signal.dt <- rbindlist(grid.signal.dt.list))
if(require(animint2)){
  ggplot()+
    geom_point(aes(
      x, y),
      data=reg.data)+
    geom_line(aes(
      x, y),
      color="red",
      size=2,
      data=grid.signal.dt)+
    facet_grid(signal ~ difficulty, labeller=label_both)
}

reg_size_cv <- mlr3resampling::ResamplingVariableSizeTrainCV$new()
n.train.sizes <- 18
reg_size_cv$param_set$values$train_sizes <- n.train.sizes
reg_size_cv$param_set$values$folds <- n.folds
reg_size_cv$param_set$values$random_seeds <- 1
reg_size_cv$instantiate(reg.task.list[[1]])#required for consistent folds across tasks.

(reg.learner.list <- list(
  if(requireNamespace("rpart"))mlr3::LearnerRegrRpart$new(),
  mlr3::LearnerRegrFeatureless$new()))
(reg.bench.grid <- mlr3::benchmark_grid(
  reg.task.list,
  reg.learner.list,
  reg_size_cv))

if(FALSE){
  if(require(future))plan("multisession")
}
if(require(lgr))get_logger("mlr3")$set_threshold("warn")
(reg.bench.result <- mlr3::benchmark(
  reg.bench.grid, store_models = TRUE))

reg.bench.score <- nc::capture_first_df(
  mlr3resampling::score(reg.bench.result),
  task_id=list(
    signal=".*?",
    " ",
    difficulty=".*"))[
, signal_difficulty_Ntrain := paste(signal,difficulty,train_size),
][]
train_size_vec <- unique(reg.bench.score$train_size)

grid.task <- mlr3::TaskRegr$new("grid", grid.dt, target="y")
pred.dt.list <- list()
point.dt.list <- list()
for(score.i in 1:nrow(reg.bench.score)){
  reg.bench.row <- reg.bench.score[score.i]
  task.dt <- data.table(
    reg.bench.row$task[[1]]$data(),
    reg.bench.row$resampling[[1]]$instance$id.dt)
  set.ids <- data.table(
    Set=c("test","train")
  )[
  , data.table(row_id=reg.bench.row[[Set]][[1]])
  , by=Set]
  i.points <- set.ids[
    task.dt, on="row_id"
  ][
    is.na(Set), Set := "unused"
  ]
  point.id <- reg.bench.row[, paste(signal_difficulty_Ntrain, test.fold, algorithm)]
  point.dt.list[[point.id]] <- data.table(
    reg.bench.row[, .(signal_difficulty_Ntrain, test.fold, algorithm)],
    i.points)
  i.learner <- reg.bench.row$learner[[1]]
  pred.dt.list[[score.i]] <- data.table(
    reg.bench.row[, .(
      signal_difficulty_Ntrain, signal, difficulty, train_size, test.fold, algorithm
    )],
    as.data.table(
      i.learner$predict(grid.task)
    )[, .(x=grid.dt$x, y=response)]
  )
}
(pred.dt <- rbindlist(pred.dt.list))
(point.dt <- rbindlist(point.dt.list)[algorithm=="featureless"])
(upred <- unique(pred.dt[, .(signal_difficulty_Ntrain, signal, difficulty, train_size)]))
signal.dt <- upred[
  grid.signal.dt, on=.(signal,difficulty), allow.cartesian=TRUE]

point.dt[grepl(" 10$", signal_difficulty_Ntrain) & test.fold==1 & Set=="train"][, .SD[1:2], by=.(signal_difficulty_Ntrain)]

(reg.bench.wide <- dcast(
  reg.bench.score,
  signal + difficulty + train_size + algorithm + signal_difficulty_Ntrain ~ .,
  list(mean, sd, length, min, max),
  value.var=c("regr.mse")))
reg.bench.test <- dcast(
  reg.bench.score[, log10.mse := log10(regr.mse)],
  signal + difficulty + train_size + test.fold + signal_difficulty_Ntrain ~ algorithm,
  value.var=c("log10.mse"))
rect.x <- seq(1,log10(max.N),l=n.train.sizes)
seq.diff <- diff(rect.x)[1]/2
test.proposed <- reg.bench.test[, {
  paired <- t.test(rpart, featureless, alternative="two.sided", paired=TRUE)
  unpaired <- t.test(rpart, featureless, alternative="two.sided", paired=FALSE)
  data.table(
    mean.of.diff=paired$estimate, p.paired=paired$p.value, p.value=unpaired$p.value,
    mean.rpart=unpaired$estimate[1], mean.featureless=unpaired$estimate[2], p.unpaired=unpaired$p.value)
}, keyby=.(signal,difficulty,train_size,signal_difficulty_Ntrain)
][, `:=`(
  difference=ifelse(
    is.nan(p.value) | p.value>0.05, "not significant", "significant"),
  xmin=10^(rect.x-seq.diff),
  xmax=10^(rect.x+seq.diff)
), by=.(signal,difficulty)][]
reg.bench.join <- reg.bench.wide[
  test.proposed[, .(signal_difficulty_Ntrain,signal,difficulty,train_size,difference)],
  on=.NATURAL]
mid.x <- 10^((max(rect.x)+min(rect.x))/2)
data.color <- "grey50"
mse.limits <- c(0.04, 3.2)
mse.breaks <- c(0.05, 0.2, 0.8, 3.2)
Toff <- 1.2
Tbrk <- c(0,0.5,1)
Tbreaks <- c(Tbrk,Tbrk+Toff)
Tlabels <- c(Tbrk,Tbrk)
unused.x <- 1.1
unused.y.point <- -3
unused.y.text <- -4
Set.y.text <- 4
Tpred <- function(DT){
  if(! "Set" %in% names(DT)){
    DT <- data.table(Set=c("train","test"))[, data.table(DT), by=Set]
  }
  data.table(DT)[
  , x := ifelse(Set=="test",0,Toff)+x
  ][
    Set=="unused", `:=`(x=unused.x, y=unused.y.point)
  ][]
}
(data.sizes <- point.dt[, .(N=.N), by=.(signal_difficulty_Ntrain, test.fold, Set)])

algo.info <- rowwiseDT(
  Algorithme=, algorithm=, color=, size=,
  "sans caractères","featureless","blue",4,
  "arbre de décision", "rpart", "red", 2,
  "idéal", "ideal", "black", 1)
fr <- function(DT)DT[, let(
  Données=ifelse(difficulty=="easy", "A", "B"),
  Algorithme=factor(algorithm, algo.info$algorithm, algo.info$Algorithme)
)]
fr(reg.bench.wide)
fr(reg.bench.join)[, let(Différence=ifelse(difference=="significant", "significative", "pas significative"))]
gg <- ggplot()+
  theme_bw()+
  theme(axis.text.x=element_text(angle=60, hjust=1))+
  geom_ribbon(aes(
    train_size,
    ymin=regr.mse_mean-regr.mse_sd,
    ymax=regr.mse_mean+regr.mse_sd,
    group=Algorithme,
    fill=Algorithme),
    help=paste("Mean plus or minus one standard deviation, over", n.folds, "cross-validation folds."),
    color=NA,
    alpha=0.5,
    data=reg.bench.wide)+
  geom_line(aes(
    train_size, regr.mse_mean,
    group=Algorithme),
    help=paste("Mean over", n.folds, "cross-validation folds."),
    color="grey",
    showSelected="Algorithme",
    data=reg.bench.wide)+
  scale_size_manual(values=algo.info[, structure(size, names=Algorithme)])+
  scale_fill_manual(values=algo.info[, structure(color, names=Algorithme)])+
  scale_color_manual(values=c(
    significative="black",
    "pas significative"=NA))+
  geom_point(aes(
    train_size, regr.mse_mean,
    color=Différence,
    size=Algorithme,
    fill=Algorithme),
    data=reg.bench.join)+
  scale_y_log10(
    "Erreur L2 sur l’ensemble test"
  )+
  scale_x_log10(
    "Nombre d’échantillons dans l’ensemble d’entraînement",
    breaks=unique(reg.bench.join$train_size))+
  facet_wrap("Données", scales="free", labeller=label_both)
png("cv-subsets-small-large-noise-fr.png", width=12, height=3, units="in", res=200)
print(gg)
dev.off()
