library(data.table)
n.folds <- 6
N <- 1000*n.folds/(n.folds-1)
abs.x <- 3*pi
set.seed(1)
x.vec <- runif(N, -abs.x, abs.x)
str(x.vec)
reg.pattern.list <- list(
  sin=sin,
  constant=function(x)0)
standard.deviation.vec <- c(
  easy=0.1,
  hard=1.2,
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
reg_size_cv$param_set$values$train_sizes <- 9
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
    difficulty=".*"))
train_size_vec <- unique(reg.bench.score$train_size)

grid.dt <- data.table(x=seq(0,1, l=101), y=0)
grid.task <- mlr3::TaskRegr$new("grid", grid.dt, target="y")
pred.dt.list <- list()
point.dt.list <- list()
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
    reg.bench.row[, .(signal, difficulty, iteration)],
    i.points)
  i.learner <- reg.bench.row$learner[[1]]
  pred.dt.list[[score.i]] <- data.table(
    reg.bench.row[, .(
      signal, difficulty, iteration, algorithm
    )],
    as.data.table(
      i.learner$predict(grid.task)
    )[, .(x=grid.dt$x, y=response)]
  )
}
(pred.dt <- rbindlist(pred.dt.list))
(point.dt <- rbindlist(point.dt.list))

set.colors <- c(
  train="#1B9E77",
  test="#D95F02",
  unused="white")
algo.colors <- c(
  featureless="red",
  rpart="blue",
  truth="black")
algo.sizes <- c(
  truth=4,
  featureless=3,
  rpart=2)
if(require(animint2)){
  viz <- animint(
    title="Variable size train set, regression",
    pred=ggplot()+
      ggtitle("Predictions for selected train/test split")+
      theme_animint(height=400)+
      scale_fill_manual(values=set.colors)+
      geom_point(aes(
        x, y, fill=set.name),
        showSelected="iteration",
        size=3,
        shape=21,
        data=point.dt)+
      scale_size_manual(values=algo.sizes)+
      scale_color_manual(values=algo.colors)+
      geom_line(aes(
        x, signal, color=algorithm, size=algorithm),
        data=reg.data)+
      geom_line(aes(
        x, y,
        color=algorithm,
        size=algorithm,
        group=paste(algorithm, iteration)),
        showSelected="iteration",
        data=pred.dt)+
      facet_grid(
        task_id ~ .,
        labeller=label_both),
    err=ggplot()+
      ggtitle("Test error for each split")+
      theme_animint(width=500)+
      theme(
        panel.margin=grid::unit(1, "lines"),
        legend.position="none")+
      scale_y_log10(
        "Mean squared error on test set")+
      scale_color_manual(values=algo.colors)+
      scale_x_log10(
        "Train set size",
        breaks=train_size_vec)+
      geom_line(aes(
        train_size, regr.mse,
        group=paste(algorithm, seed),
        color=algorithm),
        ##clickSelects="seed",
        ##alpha_off=0.2,
        showSelected="algorithm",
        size=4,
        data=reg.bench.score)+
      facet_grid(
        test.fold~task_id,
        labeller=label_both,
        scales="free")+
      geom_point(aes(
        train_size, regr.mse,
        color=algorithm),
        size=5,
        stroke=3,
        fill="black",
        fill_off=NA,
        showSelected=c(
          "algorithm",
          ##"seed",
          NULL),
        clickSelects="iteration",
        data=reg.bench.score),
    out.dir="cv-noise-samples",
    source="https://github.com/tdhock/2024-08-ift603-712/blob/main/cv-noise-samples.R")
  viz
}
if(FALSE){
  animint2pages(viz, "2023-12-26-train-sizes-regression")
}

(reg.bench.wide <- dcast(
  reg.bench.score,
  signal + difficulty + train_size + algorithm ~ .,
  list(mean, sd, length, min, max),
  value.var=c("regr.mse")))
reg.bench.test <- dcast(
  reg.bench.score[, log10.mse := log10(regr.mse)],
  signal + difficulty + train_size + test.fold ~ algorithm,
  value.var=c("log10.mse"))
(test.proposed <- reg.bench.test[, {
  paired <- t.test(rpart, featureless, alternative="two.sided", paired=TRUE)
  unpaired <- t.test(rpart, featureless, alternative="two.sided", paired=FALSE)
  data.table(
    mean.of.diff=paired$estimate, p.paired=paired$p.value,
    mean.proposed=unpaired$estimate[1], mean.other=unpaired$estimate[2], p.unpaired=unpaired$p.value)
}, by=.(signal,difficulty,train_size)])
test.proposed[difficulty=="hard" & signal=="sin" & train_size==1000]
ggplot()+
  geom_ribbon(aes(
    train_size,
    ymin=regr.mse_mean-regr.mse_sd,
    ymax=regr.mse_mean+regr.mse_sd,
    fill=algorithm),
    alpha=0.5,
    data=reg.bench.wide)+
  scale_size_manual(values=algo.sizes)+
  scale_fill_manual(values=algo.colors)+
  geom_point(aes(
    train_size, regr.mse_mean,
    size=algorithm,
    fill=algorithm),
    shape=21,
    data=reg.bench.wide)+
  scale_y_log10()+
  scale_x_log10()+
  facet_grid(difficulty ~ signal, labeller=label_both)

