remotes::install_github("engineerdanny/mlr3resampling@feature/pvalue-downsample")
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

SOAKED <- mlr3resampling::ResamplingSameOtherSizesCV$new()
SOAKED$param_set$values$sizes <- 0
SOAKED$param_set$values$folds <- 10
set.seed(1)
sim.meta.list <- list(
  different=rbind(
    reg.data[difficulty=="easy"][sample(.N, 400)][, Subset := "large"],
    reg.data[difficulty=="hard"][sample(.N, 200)][, Subset := "small"]
  )[, .(x,y,Subset)],
  iid_easy=reg.data[
    difficulty=="easy"
  ][sample(.N, 120)][
  , Subset := rep(c("large","large","small"), l=.N)
  ][, .(x,y,Subset)])
d_task_list <- list()
for(sim.name in names(sim.meta.list)){
  sim.i.dt <- sim.meta.list[[sim.name]]
  sub_task <- mlr3::TaskRegr$new(
    sim.name, sim.i.dt, target="y")
  sub_task$col_roles$subset <- "Subset"
  sub_task$col_roles$feature <- "x"
  d_task_list[[sim.name]] <- sub_task
}

(reg.bench.grid <- mlr3::benchmark_grid(
  d_task_list,
  mlr3::lrn("regr.rpart"),
  SOAKED))

if(FALSE){
  if(require(future))plan("multisession")
}
if(require(lgr))get_logger("mlr3")$set_threshold("warn")
(reg.bench.result <- mlr3::benchmark(
  reg.bench.grid, store_models = TRUE))

(score_dt <- mlr3resampling::score(reg.bench.result)[, .(
  test.subset, test.fold,
  train.subsets, Train_subsets, groups, n.train.groups,
  algorithm, RMSE=sqrt(regr.mse), task_id)])


plist <- mlr3resampling::pvalue_downsample(
  score_dt[task_id=="iid_easy" & test.subset=="small" & algorithm=="rpart"],
  "small",
  "rpart",
  value.var = "RMSE",
  digits = 3
)
