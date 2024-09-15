library(data.table)
n.folds <- 6
max.N <- 1000
N <- max.N*n.folds/(n.folds-1)
abs.x <- 3*pi
set.seed(1)
x.vec <- runif(N, -abs.x, abs.x)
str(x.vec)
reg.pattern.list <- list(
  sin=sin,
  constant=function(x)0)
standard.deviation.vec <- c(
  easy=0.1,
  hard=1.1,
  impossible=5)
reg.task.list <- list()
reg.data.list <- list()
norm01 <- function(z,ref=z)(z-min(ref))/(max(ref)-min(ref))
for(difficulty in names(standard.deviation.vec)){
  standard.deviation <- standard.deviation.vec[[difficulty]]
  for(signal in names(reg.pattern.list)){ # HI BASIL
    task_id <- paste(signal, difficulty)
    f <- reg.pattern.list[[signal]]
    signal.vec <- f(x.vec)
    y <- signal.vec+rnorm(N,sd=standard.deviation)
    task.dt <- data.table(
      x=norm01(x.vec),
      y = norm01(y))
    reg.data.list[[paste(difficulty, task_id)]] <- data.table(
      difficulty,
      signal,
      task_id,
      algorithm="signal",
      signal.vec=norm01(signal.vec,y),
      task.dt)
    reg.task.list[[paste(difficulty, task_id)]] <- mlr3::TaskRegr$new(
      task_id, task.dt, target="y"
    )
  }
}
(reg.data <- rbindlist(reg.data.list))

if(require(animint2)){
  ggplot()+
    geom_point(aes(
      x, y),
      data=reg.data)+
    geom_line(aes(
      x, signal.vec),
      color="red",
      data=reg.data)+
    facet_grid(signal ~ difficulty, labeller=label_both)
}

reg_size_cv <- mlr3resampling::ResamplingVariableSizeTrainCV$new()
n.train.sizes <- 9
reg_size_cv$param_set$values$train_sizes <- n.train.sizes
reg_size_cv$param_set$values$folds <- n.folds
reg_size_cv$param_set$values$random_seeds <- 1
reg_size_cv

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

grid.dt <- data.table(x=seq(0,1, l=101), y=0)
grid.task <- mlr3::TaskRegr$new("grid", grid.dt, target="y")
pred.dt.list <- list()
point.dt.list <- list()
signal.dt.list <- list()
for(score.i in 1:nrow(reg.bench.score)){
  reg.bench.row <- reg.bench.score[score.i]
  task.dt <- data.table(
    reg.bench.row$task[[1]]$data(),
    reg.bench.row$resampling[[1]]$instance$id.dt)
  set.ids <- data.table(
    set.name=c("test","train")
  )[
  , data.table(row_id=reg.bench.row[[set.name]][[1]])
  , by=set.name]
  i.points <- set.ids[
    task.dt, on="row_id"
  ][
    is.na(set.name), set.name := "unused"
  ]
  point.dt.list[[score.i]] <- data.table(
    reg.bench.row[, .(signal_difficulty_Ntrain, test.fold)],
    i.points)
  i.learner <- reg.bench.row$learner[[1]]
  pred.dt.list[[score.i]] <- data.table(
    reg.bench.row[, .(
      signal_difficulty_Ntrain, test.fold, algorithm
    )],
    as.data.table(
      i.learner$predict(grid.task)
    )[, .(x=grid.dt$x, y=response)]
  )
  signal.dt.list[[score.i]] <- reg.data[reg.bench.row, on=.(signal, difficulty)]  
}
(pred.dt <- rbindlist(pred.dt.list))
(point.dt <- rbindlist(point.dt.list))
(signal.dt <- rbindlist(signal.dt.list))

algo.colors <- c(
  featureless="red",
  rpart="blue",
  signal="grey50")
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
    mean.of.diff=paired$estimate, p.paired=paired$p.value,
    mean.rpart=unpaired$estimate[1], mean.featureless=unpaired$estimate[2], p.unpaired=unpaired$p.value)
}, keyby=.(signal,difficulty,train_size,signal_difficulty_Ntrain)
][, `:=`(
  difference=ifelse(
    is.nan(p.paired) | p.paired>0.05, "not significant", "significant"),
  xmin=10^(rect.x-seq.diff),
  xmax=10^(rect.x+seq.diff)
), by=.(signal,difficulty)][]
test.proposed[difficulty=="hard" & signal=="sin"]
reg.bench.join <- reg.bench.wide[
  test.proposed[, .(signal_difficulty_Ntrain,signal,difficulty,train_size,difference)],
  on=.NATURAL]
mid.x <- 10^((max(rect.x)+min(rect.x))/2)
lim <- c(0.005, 0.5)
animint(
  title="Samples required to learn non-trivial regression model",
  overview=ggplot()+
    ggtitle("Select signal, difficulty, Ntrain")+
    theme_bw()+
    theme_animint(width=600, height=300)+
    geom_rect(aes(
      xmin=xmin, xmax=xmax,
      ymin=0, ymax=Inf),
      alpha=0.1,
      fill="black",
      color=NA,
      clickSelects="signal_difficulty_Ntrain",
      data=test.proposed)+
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
      xend=train_size, yend=10^mean.featureless),
      showSelected="signal_difficulty_Ntrain",
      size=3,
      alpha=0.5,
      data=test.proposed)+
    geom_text(aes(
      train_size, 10^pmax(mean.rpart,mean.featureless)*1.2,
      hjust=ifelse(train_size<mid.x, 0, 1),
      label=fcase(
        p.paired<0.0001, "p<0.0001",
        is.nan(p.paired), "Diff=0",
        default=sprintf("p=%.4f", p.paired))),
      showSelected="signal_difficulty_Ntrain",
      data=test.proposed)+
    ## geom_point(aes(
    ##   10^(log10(train_size)+seq.diff), regr.mse,
    ##   fill=algorithm),
    ##   showSelected="signal_difficulty_Ntrain",
    ##   clickSelects="test.fold",
    ##   color=NA,
    ##   data=reg.bench.score)+
    scale_y_log10(
      "Mean Squared Error (log scale)",
      limits=c(0.007, 0.12))+
    scale_x_log10()+
    facet_grid(signal ~ difficulty, labeller=label_both),
  scatter=ggplot()+
    ggtitle("MSE for selected")+
    theme_bw()+
    theme_animint(width=400, height=300)+
    theme(legend.position="none")+
    coord_equal(xlim=lim, ylim=lim)+
    scale_x_log10(
      "featureless")+
    scale_y_log10(
      "rpart")+
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
      10^featureless, 10^rpart, tooltip=sprintf(
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
      "Mean Squared Error (log scale)")+
    scale_color_manual(values=algo.colors)+
    geom_point(aes(
      regr.mse, algorithm,
      color=algorithm,
      tooltip=sprintf(
        "%s fold %d MSE=%.3f", algorithm, test.fold, regr.mse)),
      showSelected=c("algorithm","signal_difficulty_Ntrain"),
      clickSelects="test.fold",
      alpha=0.7,
      size=5,
      data=reg.bench.score),
  pred=ggplot()+
    ggtitle("Predictions for selected train/test split")+
    theme_bw()+
    theme_animint(height=300, width=1000)+
    geom_point(aes(
      x, y),
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      size=3,
      fill="white",
      color="black",
      data=point.dt[set.name!="unused"][, Set := set.name])+
    scale_size_manual(values=algo.sizes)+
    scale_color_manual(values=algo.colors)+
    geom_line(aes(
      x, signal.vec, color=algorithm, size=algorithm),
      showSelected=c("signal_difficulty_Ntrain"),
      data=signal.dt)+
    geom_line(aes(
      x, y,
      color=algorithm,
      size=algorithm,
      group=algorithm),
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      data=pred.dt)+
    facet_grid(
      . ~ Set,
      labeller=label_both),
  out.dir="cv-noise-samples",
  source="https://github.com/tdhock/2024-08-ift603-712/blob/main/cv-noise-samples.R",
  first=list(
    signal_difficulty_Ntrain="sin easy 1000")
)
if(FALSE){
  animint2pages(viz, "2024-09-15-train-sizes-regression")
}

