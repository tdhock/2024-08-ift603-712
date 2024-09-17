library(data.table)
n.folds <- 6
max.N <- 1000
N <- max.N*n.folds/(n.folds-1)
abs.x <- 3*pi
set.seed(1)
norm01 <- function(z,ref=z)(z-min(ref))/(max(ref)-min(ref))
(grid.dt <- data.table(
  raw=seq(-abs.x,abs.x, l=201),
  y=0 #for mlr3
)[, x := norm01(raw)][])
x.vec <- runif(N, -abs.x, abs.x)
str(x.vec)
reg.pattern.list <- list(
  sin=sin,
  constant=function(x)0)
standard.deviation.vec <- c(
  easy=0.4,
  hard=1.1,
  impossible=5)
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
      x=norm01(x.vec,grid.dt$raw),
      y = norm01(y))
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
      y=norm01(f(grid.dt$raw),y)
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
n.train.sizes <- 9
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

algo.colors <- c(
  featureless="red",
  rpart="deepskyblue",
  ideal="black")
algo.sizes <- c(
  truth=6,
  featureless=4,
  rpart=2)
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
if(FALSE){#bug in R?
  reg.bench.test[difficulty=="hard" & train_size==1000 & signal=="constant", t.test(featureless, rpart, paired=TRUE)]
  reg.bench.test[difficulty=="hard" & train_size==1000 & signal=="constant", t.test(featureless, rpart, paired=FALSE)]
  dput(data.frame(reg.bench.test[difficulty=="hard" & train_size==1000 & signal=="constant", .(err1=featureless, err2=rpart)]), control="digits17")
  err1 = c(-1.6076199373862132, -1.658521185520103, -1.6549424312339873, -1.5887767975086149, -1.634129577540383, -1.7442711937982249)
  err2 = c(-1.6076199373862132, -1.6585211855201032, -1.6549424312339875, -1.5887767975086149, -1.6341295775403832, -1.7442711937982252)
  t.test(err1,err2,paired=TRUE)
  t.test(err1,err2,paired=FALSE)
}
test.proposed[difficulty=="hard" & train_size==1000]
reg.bench.join <- reg.bench.wide[
  test.proposed[, .(signal_difficulty_Ntrain,signal,difficulty,train_size,difference)],
  on=.NATURAL]
mid.x <- 10^((max(rect.x)+min(rect.x))/2)
data.color <- "grey50"
mse.limits <- c(0.01, 0.045)
mse.breaks <- c(0.01,0.02,0.04)
Toff <- 1.2
Tbrk <- c(0,0.5,1)
Tbreaks <- c(Tbrk,Tbrk+Toff)
Tlabels <- c(Tbrk,Tbrk)
unused.x <- 1.1
unused.y.point <- 0.1
unused.y.text <- 0
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
viz <- animint(
  title="Samples required to learn non-trivial regression model",
  overview=ggplot()+
    ggtitle("Select signal, difficulty, Ntrain")+
    theme_bw()+
    theme_animint(width=600, height=300)+
    geom_ribbon(aes(
      train_size,
      ymin=regr.mse_mean-regr.mse_sd,
      ymax=regr.mse_mean+regr.mse_sd,
      group=algorithm,
      fill=algorithm),
      color=NA,
      alpha=0.5,
      data=reg.bench.wide)+
    scale_size_manual(values=algo.sizes)+
    scale_fill_manual(values=algo.colors)+
    scale_color_manual(values=c(
      significant="black",
      "not significant"=NA))+
    geom_point(aes(
      train_size, regr.mse_mean,
      color=difference,
      size=algorithm,
      fill=algorithm),
      data=reg.bench.join)+
    geom_segment(aes(
      train_size, 10^mean.rpart,
      key=1,
      xend=train_size, yend=10^mean.featureless),
      showSelected="signal_difficulty_Ntrain",
      size=3,
      alpha=0.5,
      data=test.proposed)+
    geom_text(aes(
      train_size, 10^pmax(mean.rpart,mean.featureless)*1.1,
      key=1,
      hjust=ifelse(train_size<mid.x, 0, 1),
      label=fcase(
        p.value<0.0001, "p<0.0001",
        is.nan(p.value), "Diff=0",
        default=sprintf("p=%.4f", p.value))),
      showSelected="signal_difficulty_Ntrain",
      data=test.proposed)+
    geom_rect(aes(
      xmin=xmin, xmax=xmax,
      ymin=0, ymax=Inf),
      alpha=0.1,
      fill="black",
      color=NA,
      clickSelects="signal_difficulty_Ntrain",
      data=test.proposed)+
    scale_y_log10(
      "Mean Squared Error (log scale)",
      limits=mse.limits,
      breaks=mse.breaks
    )+
    scale_x_log10(
      "Ntrain = Number of samples in train set (log scale)")+
    facet_grid(signal ~ difficulty, labeller=label_both),
  scatter=ggplot()+
    ggtitle("MSE for selected")+
    theme_bw()+
    theme_animint(width=300, height=300)+
    theme(legend.position="none")+
    coord_equal(xlim=mse.limits, ylim=mse.limits)+
    scale_x_log10(
      "featureless (log scale)",
      breaks=mse.breaks)+
    scale_y_log10(
      "rpart (log scale)",
      breaks=mse.breaks)+
    geom_abline(aes(
      slope=slope, intercept=intercept),
      color="grey50",
      data=data.table(slope=1, intercept=0))+
    geom_segment(aes(
      x, y, xend=xend, yend=yend, color=algorithm),
      data=rbind(
        data.table(x=0, y=0, xend=0, yend=Inf, algorithm="rpart"),
        data.table(x=0, y=0, xend=Inf, yend=0, algorithm="featureless")),
      alpha=0.5,
      showSelected="algorithm",
      size=5)+
    scale_color_manual(values=algo.colors)+
    geom_point(aes(
      10^featureless, 10^rpart,
      key=test.fold,
      tooltip=sprintf(
        "fold %d featureless=%.3f rpart=%.3f", test.fold, featureless, rpart)),
      showSelected="signal_difficulty_Ntrain",
      clickSelects="test.fold",
      size=5,
      alpha=0.7,
      data=reg.bench.test),
  details=ggplot()+
    ggtitle("MSE for selected")+
    theme_bw()+
    theme_animint(width=1000, height=120)+
    theme(legend.position="none")+
    scale_y_discrete("Algo")+
    scale_x_log10(
      "Mean Squared Error (log scale)",
      limits=mse.limits,
      breaks=mse.breaks)+
    scale_color_manual(values=algo.colors)+
    geom_point(aes(
      regr.mse, algorithm,
      key=paste0(algorithm, test.fold),
      color=algorithm,
      tooltip=sprintf(
        "%s fold %d MSE=%.3f", algorithm, test.fold, regr.mse)),
      showSelected=c("algorithm","signal_difficulty_Ntrain"),
      clickSelects="test.fold",
      alpha=0.7,
      size=5,
      data=reg.bench.score),
  ## pred=ggplot()+
  ##   ggtitle("Predictions for selected train/test split")+
  ##   theme_bw()+
  ##   theme_animint(height=300, width=900)+
  ##   geom_point(aes(
  ##     x, y,
  ##     key=paste(signal_difficulty_Ntrain, row_id)),
  ##     showSelected=c("signal_difficulty_Ntrain","test.fold"),
  ##     size=3,
  ##     fill="white",
  ##     color=data.color,
  ##     data=point.dt[Set!="unused"])+
  ##   scale_x_continuous("x = feature/input")+
  ##   scale_y_continuous("y = label/output")+
  ##   scale_size_manual(values=algo.sizes)+
  ##   scale_color_manual(values=algo.colors)+
  ##   geom_line(aes(
  ##     x, y,
  ##     key=algorithm,
  ##     color=algorithm,
  ##     size=algorithm,
  ##     group=algorithm),
  ##     showSelected=c("signal_difficulty_Ntrain","test.fold"),
  ##     data=pred.dt)+
  ##   geom_line(aes(
  ##     x, y,
  ##     key=algorithm,
  ##     color=algorithm,
  ##     size=algorithm),
  ##     showSelected=c("signal_difficulty_Ntrain"),
  ##     data=signal.dt)+
  ##   geom_text(aes(
  ##     0, 0.98,
  ##     key=Set,
  ##     label=paste0("N",Set,"=",N)),
  ##     hjust=0,
  ##     data=data.sizes,
  ##     color=data.color,
  ##     showSelected="signal_difficulty_Ntrain")+
  ##   facet_grid(
  ##     . ~ Set,
  ##     labeller=label_both),
  pred=ggplot()+
    ggtitle("Predictions for selected train/test split")+
    theme_bw()+
    theme_animint(height=300, width=900)+
    geom_point(aes(
      x, y,
      key=row_id),
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      size=3,
      fill="white",
      color=data.color,
      data=Tpred(point.dt))+
    scale_x_continuous(
      "x = feature/input",
      labels=Tlabels,
      breaks=Tbreaks)+
    scale_y_continuous("y = label/output")+
    geom_line(aes(
      x, y,
      key=paste(algorithm,Set),
      color=algorithm,
      size=algorithm,
      group=paste(algorithm,Set)),
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      data=Tpred(pred.dt))+
    geom_line(aes(
      x, y,
      key=Set,
      group=Set,
      color=algorithm,
      size=algorithm),
      showSelected=c("signal_difficulty_Ntrain"),
      data=Tpred(signal.dt))+
    geom_text(aes(
      x, ifelse(Set=="unused", unused.y.text, 0.98),
      hjust=ifelse(Set=="unused", 0.5, 0),
      key=Set,
      label=paste0(Set," set N=",N)),
      data=Tpred(data.sizes[, x := 0]),
      color=data.color,
      showSelected="signal_difficulty_Ntrain")+
    scale_size_manual(values=algo.sizes)+
    scale_color_manual(values=algo.colors),
  out.dir="cv-noise-samples",
  source="https://github.com/tdhock/2024-08-ift603-712/blob/main/cv-noise-samples.R",
  duration=list(
    test.fold=1000,
    signal_difficulty_Ntrain=1000),
  first=list(
    signal_difficulty_Ntrain="sin easy 1000")
)
if(FALSE){
  animint2pages(viz, "2024-09-15-K-fold-CV-train-sizes-regression")
  animint2pages(viz, "2024-09-16-K-fold-CV-train-sizes-regression")
}
viz

