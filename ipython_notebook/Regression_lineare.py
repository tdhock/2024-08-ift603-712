## conda install python=3.10 pandas plotnine
import pandas as pd
import numpy as np
reg_df = pd.read_csv("Regression_lineare.csv")
set_dict = {"train":{
    "y": reg_df["y"].to_numpy(),
    "X": reg_df.iloc[:, 1:].to_numpy()
}}
Nrow, Ncol = set_dict["train"]["X"].shape

# first split
fold_vec = np.tile(np.arange(3),Nrow)[:Nrow]
valid_fold=1
set_vec = np.where(fold_vec==valid_fold, "validation", "subtrain")
for set_name in "validation", "subtrain":
    is_set = set_vec == set_name
    set_dict[set_name] = {
        "X":set_dict["train"]["X"][is_set,:],
        "y":set_dict["train"]["y"][is_set]
    }
{set_name:D['X'].shape for set_name,D in set_dict.items()}

# then grad desc.
weight_vec = np.repeat(0.0, Ncol)
max_iterations = 1000
step_size = 0.001
err_df_list = []
def get_diff(xy_dict):
    pred = np.matmul(xy_dict["X"], weight_vec)
    return pred - xy_dict["y"]
for iteration in range(max_iterations):
    subtrain_diff = get_diff(set_dict["subtrain"])
    grad_vec = np.matmul(set_dict["subtrain"]["X"].T, subtrain_diff)
    weight_vec -= step_size * grad_vec
    for set_name in "subtrain", "validation":
        set_diff = get_diff(set_dict[set_name])
        set_err = set_diff **2
        err_df_list.append(pd.DataFrame({
            "iteration":[iteration],
            "set_name":[set_name],
            "mean_square_error":[set_err.mean()]
        }))
err_df = pd.concat(err_df_list)

import plotnine as p9
show_df = err_df.query("iteration < 10000")
gg = p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="iteration",
            y="mean_square_error",
            color="set_name"
        ),
        data = show_df)
#p9.facet_grid("set_name ~ .", labeller="label_both", scales="free")
gg.show()#dans une fenetre
gg.save("Regression_lineare.png", width=10, height=5, dpi=200)

err_df.query("set_name == 'validation'").mean_square_error.argmin()
