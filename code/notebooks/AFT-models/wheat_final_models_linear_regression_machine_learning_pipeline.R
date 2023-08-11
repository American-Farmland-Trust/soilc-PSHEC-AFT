#' ---
#' title: "grouped_states_model_pipeline.Rmd"
#' author: "ML"
#' date: "2023-06-05"
#' output: html_document
#' editor_options: 
#'   chunk_output_type: console
#' ---
#' 
#' ### Use the tidymodels frame work to compare different preprocessing&model combinations
#' 
#' # Setup
## ----setup-----------------------------------------------------------------------------------------------------------------------------------------
library(tidymodels)  # Includes the workflows package
tidymodels_prefer() # resolve the name conflicts

library(ranger)
library(multilevelmod)
library(lme4)
library(rules)
library(finetune)
library(DALEXtra)
library(ggpmisc)


#' 
#' # Read in data and filter by rainfed states
#' 
## ----data------------------------------------------------------------------------------------------------------------------------------------------

comet_sum <- read.csv('data/soil/Comet_data/Comet-som-county-max-sum.csv')
comet_sum$GEOID <- formatC(comet_sum$GEOID, width = 5, format = 'd' ,flag = '0')

all_data <- readRDS('data/AFT-data/wheat_all_data_n15_om10_AFT.rds')

rainfed <- all_data %>%
  filter(irrigated == 'rainfed') 

#levels(as.factor(rainfed$state_alpha))

# "AR" "CO" "GA" "ID" "IL" "IN" "KS" "KY" "MD" "MI" "MO" "MS" "MT" "NC" "ND" "NE" "NJ" "NY" "OH" "OK" "OR" "PA" "SC" "SD" "TN" "TX" "VA" "WA" "WI" "WV"

# create a list of state (/group of state) names

state_id <- list(
  all = c("AR", "CO", "GA", "ID", "IL", "IN", "KS", "KY", "MD", "MI", "MO", "MS", "MT", "NC", "ND", "NE", "NJ", "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TN", "TX", "VA", "WA", "WI", "WV"),
  group_23 = c("AR", "CO", "GA", "ID", "KS", "KY", "MD", "MI", "MO", "MS", "MT", "NC", "ND", "NE", "NJ", "NY", "OK", "OR", "PA", "SC", "SD", "TN", "TX", "VA", "WA", "WI", "WV"),
  IL_IN_OH = c('IL','IN','OH')
)


rm(all_data)


#' 
#' 
#' # Start the tidymodels workflow
#' 
#' ## 1. Split the data using the default 3:1 ratio of training-to-test and resample the training set using five repeats of 10-fold cross-validation
## ----splitdata-------------------------------------------------------------------------------------------------------------------------------------

# create a loop to create files for each individual state

for (i in names(state_id)) {
  
  data <- rainfed %>% 
    filter(state_alpha %in% state_id[[i]]) %>%
    droplevels()
  
  #--------individual state----------------#
  set.seed(1501)
  wheat_split <- initial_split(data)
  wheat_train <- training(wheat_split)
  wheat_test  <- testing(wheat_split)
  
  set.seed(1502)
  wheat_folds <- 
    vfold_cv(wheat_train, repeats = 5)
  
  
  #' 
  #' ## 2. create the data preprocessing recipe
  #' 
  ## ----recipe----------------------------------------------------------------------------------------------------------------------------------------
  # the basic recipe contains all available soil variables
  basic_rec <- 
    recipe(Yield_decomp_add ~ 
             ssurgo_om_mean + 
             ssurgo_clay_mean + ssurgo_silt_mean + ssurgo_sand_mean + 
             ssurgo_awc_mean + ssurgo_aws_mean + ssurgo_fifteenbar_mean +
             ssurgo_cec_mean +ssurgo_h +  GEOID + state_alpha, data =wheat_train) %>%
    add_role(ssurgo_om_mean, new_role = 'om') %>% # add the role of om so that it won't be removed in the following steps
    update_role(GEOID, new_role = 'ID')
  
  
  #' 
  #' 2.2 Specific recipes for different models
  ## ----spec------------------------------------------------------------------------------------------------------------------------------------------
  library(embed)
  
  # selected less correlated variables and two-way interaction terms are the predictors; state was converted to dummy variables
  lasso_corr_d_int2 <- 
    basic_rec %>%
    step_corr(all_numeric_predictors(),-has_role('om'), threshold = 0.7) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_interact(~all_numeric_predictors()^2, sep = '*') %>%
    step_dummy(state_alpha) %>%
    step_interact(~all_numeric_predictors():starts_with('state_alpha_'), sep = '*') 
  # note: used corr = 0.7 as the cut off because a study showed that collinearity becomes a more serious problem (inflating the variance of estimated regression coefficients, and therefore not necessarily finding the 'significant' ones correctly) at r>0.7 (reference:  https://doi.org/10.1111/j.1600-0587.2012.07348.x)
  
  
  # selected less correlated variables and two-way interaction terms are the predictors; state was convert into a single set of scores derived from a generalized linear mixed model. lmer(outcome ~ 1 + (1 | predictor)
  lasso_corr_m_int2 <- 
    basic_rec %>%
    step_corr(all_numeric_predictors(),-has_role('om'), threshold = 0.7) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_interact(~all_numeric_predictors()^2, sep = '*') %>%
    step_lencode_mixed(state_alpha, outcome = vars(Yield_decomp_add))
  
  # selected less correlated variables and three-way interaction terms are the predictors; state was converted to dummy variables
  lasso_corr_d_int3 <- 
    basic_rec %>%
    step_corr(all_numeric_predictors(),-has_role('om'), threshold = 0.7) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_interact(~all_numeric_predictors()^3, sep = '*') %>%
    step_dummy(state_alpha) %>%
    step_interact(~all_numeric_predictors():starts_with('state_alpha_'), sep = '*') 
  # note: arbitrarily used up to three-way interactions because the higher level of interactions do not usually add extra values
  
  # selected less correlated variables and three-way interaction terms are the predictors; state was convert into a single set of scores derived from a generalized linear mixed model. lmer(outcome ~ 1 + (1 | predictor)
  lasso_corr_m_int3 <- 
    basic_rec %>%
    step_corr(all_numeric_predictors(),-has_role('om'), threshold = 0.7) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_interact(~all_numeric_predictors()^3, sep = '*') %>%
    step_lencode_mixed(state_alpha, outcome = vars(Yield_decomp_add))
  
  normalize_rec <-
    basic_rec %>%
    step_normalize(all_numeric_predictors()) 
  
  norm_dum_rec <-
    basic_rec %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(state_alpha)
  
  
  #' 
  #' # create model specification 
  #' 
  ## ----model-----------------------------------------------------------------------------------------------------------------------------------------
  
  library(ranger)
  library(xgboost)
  library(Cubist)
  
  #Note: used lasso model to select variables with penalty: lasso model was recommended over stepwise models; and the final predictive model also does a better job than the stepwise model; 
  #see discussions here:https://stats.stackexchange.com/questions/20836/algorithms-for-automatic-model-selection/20856#20856
  lasso_model <- linear_reg(penalty = tune(),mixture = 1) %>%
    set_mode('regression') %>%
    set_engine('glmnet') # In the glmnet model, mixture = 1 is a pure lasso model while mixture = 0 indicates that ridge regression is being used.
  
  # cart and rf models can also be used: these are non-linear regressions
  cart_model <- 
    decision_tree(cost_complexity = tune(), min_n = tune()) %>% 
    set_engine("rpart") %>% 
    set_mode("regression") # a regression tree model 
  
  rf_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
    set_engine("ranger") %>% 
    set_mode("regression")
  
  nnet_model <- 
    mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
    set_engine("nnet", MaxNWts = 2600) %>% 
    set_mode("regression")
  
  # note: The analysis in M. Kuhn and Johnson (2013) specifies that the neural network should have up to 27 hidden units in the layer. The extract_parameter_set_dials() function extracts the parameter set, which we modify to have the correct parameter range:https://www.tmwr.org/workflow-sets.html
  nnet_param <- 
    nnet_model %>% 
    extract_parameter_set_dials() %>% 
    update(hidden_units = hidden_units(c(1, 27)))
  
  xgb_model <- 
    boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
               min_n = tune(), sample_size = tune(), trees = tune()) %>% 
    set_engine("xgboost") %>% 
    set_mode("regression")
  
  cubist_model <- 
    cubist_rules(committees = tune(), neighbors = tune()) %>% 
    set_engine("Cubist") 
  
  gam_model <- gen_additive_mod(adjust_deg_free = tune()) %>%
    set_mode('regression') %>%
    set_engine('mgcv')
  
  
  #' 
  #' ## create model workflow_set
  #' 
  ## ----workflow--------------------------------------------------------------------------------------------------------------------------------------
  library(multilevelmod)
  library(lme4)
  library(rules)
  
  # add machine learning flows
  all_workflows <- 
    workflow_set(
      preproc = list(lasso_corr_d_2 = lasso_corr_d_int2,lasso_corr_m_2 = lasso_corr_m_int2, 
                     lasso_corr_d_3 = lasso_corr_d_int3,lass_corr_m_3 = lasso_corr_m_int3,
                     cart = normalize_rec, 
                     rf = normalize_rec,
                     nnet = normalize_rec,
                     xgb = norm_dum_rec,
                     cubist = normalize_rec,
                     gam_om = norm_dum_rec,
                     gam = norm_dum_rec
      ),
      models = list(lasso_model, lasso_model, lasso_model, lasso_model,
                    cart_model,
                    rf_model,
                    nnet_model,
                    xgb_model,
                    cubist_model,
                    gam_model,
                    gam_model
      ),
      cross = FALSE
    )

  # update all_workflows to integrate special formula for GAM model 
  if (i == 'all') {
    
    all_workflows <- all_workflows %>%
      option_add(param_info = nnet_param, id = "nnet_mlp") %>%
      update_workflow_model(id = 'gam_om_gen_additive_mod_10', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              ssurgo_clay_mean + ssurgo_silt_mean + ssurgo_sand_mean + 
                              ssurgo_awc_mean + ssurgo_aws_mean + ssurgo_fifteenbar_mean +
                              ssurgo_cec_mean +ssurgo_h + 
                              state_alpha_CO + state_alpha_GA + state_alpha_ID +
                              state_alpha_IL + state_alpha_IN + state_alpha_KS +        
                              state_alpha_KY + state_alpha_MD + state_alpha_MI +        
                              state_alpha_MO + state_alpha_MS + state_alpha_MT + state_alpha_NC + state_alpha_ND + state_alpha_NE + state_alpha_NJ +        
                              state_alpha_NY + state_alpha_OH + state_alpha_OK + state_alpha_OR + state_alpha_PA + state_alpha_SC +      
                              state_alpha_SD + state_alpha_TN + state_alpha_TX + state_alpha_VA + state_alpha_WA + state_alpha_WI +        
                              state_alpha_WV) %>%
      update_workflow_model(id = 'gam_gen_additive_mod_11', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              s(ssurgo_clay_mean) + s(ssurgo_silt_mean) + s(ssurgo_sand_mean) + 
                              s(ssurgo_awc_mean) + s(ssurgo_aws_mean) + s(ssurgo_fifteenbar_mean) +
                              s(ssurgo_cec_mean) + s(ssurgo_h) +
                              state_alpha_CO + state_alpha_GA + state_alpha_ID +
                              state_alpha_IL + state_alpha_IN + state_alpha_KS +        
                              state_alpha_KY + state_alpha_MD + state_alpha_MI +        
                              state_alpha_MO + state_alpha_MS + state_alpha_MT + state_alpha_NC + state_alpha_ND + state_alpha_NE + state_alpha_NJ +        
                              state_alpha_NY + state_alpha_OH + state_alpha_OK + state_alpha_OR + state_alpha_PA + state_alpha_SC +      
                              state_alpha_SD + state_alpha_TN + state_alpha_TX + state_alpha_VA + state_alpha_WA + state_alpha_WI +        
                              state_alpha_WV)
    
  } else if (i == 'group_23') {
    
    all_workflows <- all_workflows %>%
      option_add(param_info = nnet_param, id = "nnet_mlp") %>%
      update_workflow_model(id = 'gam_om_gen_additive_mod_10', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              ssurgo_clay_mean + ssurgo_silt_mean + ssurgo_sand_mean + 
                              ssurgo_awc_mean + ssurgo_aws_mean + ssurgo_fifteenbar_mean +
                              ssurgo_cec_mean +ssurgo_h + 
                              state_alpha_CO + state_alpha_GA + state_alpha_ID +
                              #state_alpha_IL + state_alpha_IN + 
                              state_alpha_KS +        
                              state_alpha_KY + state_alpha_MD + state_alpha_MI +        
                              state_alpha_MO + state_alpha_MS + state_alpha_MT + state_alpha_NC + state_alpha_ND + state_alpha_NE + state_alpha_NJ +        
                              state_alpha_NY + 
                              #state_alpha_OH + 
                              state_alpha_OK + state_alpha_OR + state_alpha_PA + state_alpha_SC +      
                              state_alpha_SD + state_alpha_TN + state_alpha_TX + state_alpha_VA + state_alpha_WA + state_alpha_WI +        
                              state_alpha_WV) %>%
      update_workflow_model(id = 'gam_gen_additive_mod_11', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              s(ssurgo_clay_mean) + s(ssurgo_silt_mean) + s(ssurgo_sand_mean) + 
                              s(ssurgo_awc_mean) + s(ssurgo_aws_mean) + s(ssurgo_fifteenbar_mean) +
                              s(ssurgo_cec_mean) + s(ssurgo_h) +
                              state_alpha_CO + state_alpha_GA + state_alpha_ID +
                              #state_alpha_IL + state_alpha_IN + 
                              state_alpha_KS +        
                              state_alpha_KY + state_alpha_MD + state_alpha_MI +        
                              state_alpha_MO + state_alpha_MS + state_alpha_MT + state_alpha_NC + state_alpha_ND + state_alpha_NE + state_alpha_NJ +        
                              state_alpha_NY + 
                              #state_alpha_OH + 
                              state_alpha_OK + state_alpha_OR + state_alpha_PA + state_alpha_SC +      
                              state_alpha_SD + state_alpha_TN + state_alpha_TX + state_alpha_VA + state_alpha_WA + state_alpha_WI +        
                              state_alpha_WV)
  } else {
    
    all_workflows <- all_workflows %>%
      option_add(param_info = nnet_param, id = "nnet_mlp") %>%
      update_workflow_model(id = 'gam_om_gen_additive_mod_10', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              ssurgo_clay_mean + ssurgo_silt_mean + ssurgo_sand_mean + 
                              ssurgo_awc_mean + ssurgo_aws_mean + ssurgo_fifteenbar_mean +
                              ssurgo_cec_mean +ssurgo_h + 
                              state_alpha_IN + state_alpha_OH) %>% #IL will not be in the dummy variables
      update_workflow_model(id = 'gam_gen_additive_mod_11', spec = gam_model,
                            formula = Yield_decomp_add ~ s(ssurgo_om_mean) + 
                              s(ssurgo_clay_mean) + s(ssurgo_silt_mean) + s(ssurgo_sand_mean) + 
                              s(ssurgo_awc_mean) + s(ssurgo_aws_mean) + s(ssurgo_fifteenbar_mean) +
                              s(ssurgo_cec_mean) + s(ssurgo_h) +
                              state_alpha_IN + state_alpha_OH)
    
  }
  
  #' 
  #' ## Run all workflows
  #' 
  ## ----run-------------------------------------------------------------------------------------------------------------------------------------------
  library(finetune)
  # to effectively screen a large set of models using the race method:https://www.tmwr.org/grid-search.html#racing
  
  race_ctrl <-
    control_race(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
    )
  
  race_results <-
    all_workflows %>%
    workflow_map(
      "tune_race_anova",
      seed = 1503,
      resamples = wheat_folds,
      grid = 25,
      control = race_ctrl
    )
  
  # show the results of models
  model_rmse <- autoplot(
    race_results,
    rank_metric = "rmse",  
    metric = "rmse",       
    select_best = TRUE    
  ) +
    geom_text(aes(y = mean + 1/2, label = wflow_id), angle = 90, hjust = 1) +
    theme(legend.position = "none")
  
  ggsave(model_rmse, filename = paste0('data/soil/Model_selection/wheat/', i , '_ML_model_selection.png'))
  
  # show metrics
  best_wf<-race_results %>%
    rank_results(select_best = TRUE) %>%
    filter(.metric == "rmse") %>%
    filter(rank == 1) %>%
    droplevels() 
  
  best_wf_name <- best_wf$wflow_id
  
  #' 
  #' ## Finalize the final model
  #' 
  ## ----final-----------------------------------------------------------------------------------------------------------------------------------------
  best_results <- 
    race_results %>% 
    extract_workflow_set_result(best_wf_name) %>% 
    select_best(metric = "rmse")
  #best_results
  
  rf_test_results <- 
    race_results %>% 
    extract_workflow(best_wf_name) %>% 
    finalize_workflow(best_results) %>% 
    last_fit(split = wheat_split)
  
  #rf_test_metrics <- collect_metrics(rf_test_results)
  
  # plot test set prediction results 
  rf_model_plot <- rf_test_results %>% 
    collect_predictions() %>% 
    ggplot(aes(x = Yield_decomp_add, y = .pred)) + 
    geom_smooth(method = 'lm') +
    geom_abline(color = "gray50", lty = 2) + 
    geom_point(alpha = 0.5) + 
    coord_obs_pred() + 
    labs(x = "observed", y = "predicted")
  
  ggsave(rf_model_plot, filename = paste0('data/soil/Model_selection/wheat/', i , '_ML_model_obs.vs.pred.png'))
  
  
  #' 
  #' 
  #' # Create the final model
  ## ----final_model-----------------------------------------------------------------------------------------------------------------------------------
  
  
  #---------------ML model--------#
  
  final_wf <- 
    race_results %>% 
    extract_workflow(best_wf_name) %>% 
    finalize_workflow(best_results) 
  
  model_final <- fit(final_wf, data)
  
  saveRDS(final_wf, file = paste0('data/soil/Model_selection/wheat/', i , '_final_workflow.rds'))
  saveRDS(model_final, file = paste0('data/soil/Model_selection/wheat/', i , '_final_model.rds'))
  
  #' 
  #' # use trained model to predict new data 
  #' 
  ## ----test_new--------------------------------------------------------------------------------------------------------------------------------------
  # all predictors
  need_features <- c('state_alpha','GEOID','ssurgo_om_mean','ssurgo_clay_mean','ssurgo_silt_mean','ssurgo_sand_mean','ssurgo_awc_mean', 'ssurgo_aws_mean', 'ssurgo_fifteenbar_mean','ssurgo_cec_mean', 'ssurgo_h')
  
  test_data <- 
    data %>% 
    select(all_of(need_features)) %>%
    unique()
  
  # meta data
  need_features.1 <- c('state_alpha','state_ansi','county_ansi','county_name','GEOID')
  
  meta_data <- 
    data %>% 
    select(all_of(need_features.1)) %>%
    unique()
  
  
  #--------------------- ML model ----------------------#
  
  rf_test_data.1 <- 
    recipe(Yield_decomp_add ~   GEOID + state_alpha +
             ssurgo_om_mean + 
             ssurgo_clay_mean + ssurgo_silt_mean + ssurgo_sand_mean + 
             ssurgo_awc_mean + ssurgo_aws_mean + ssurgo_fifteenbar_mean +
             ssurgo_cec_mean +ssurgo_h , data =data) %>%
    add_role(ssurgo_om_mean, new_role = 'om') %>%
    update_role(GEOID, new_role = 'ID') %>%
    step_normalize(all_numeric_predictors()) %>%
    prep() %>%
    bake(new_data = NULL) %>%
    group_by(GEOID) %>%
    mutate(yield_mean = mean(Yield_decomp_add)) %>%
    select(-Yield_decomp_add) %>%
    unique() %>%
    left_join(test_data[,c('GEOID','ssurgo_om_mean')], by = 'GEOID')
  
  pred_rf_county_mean <- predict(model_final, test_data) %>% rename(pred_county_mean = .pred)
  
  test_rf_res.1 <- bind_cols(rf_test_data.1, pred_rf_county_mean)  
  
  # calculate rmse 
  rmse_rf <- test_rf_res.1 %>%
    group_by(state_alpha) %>%
    mutate(state_rmse = Metrics::rmse(yield_mean, pred_county_mean), yield_min = min(yield_mean), yield_max = max(yield_mean)) %>%
    select(state_alpha, state_rmse, yield_min, yield_max) %>%
    unique()
  
  write.csv(rmse_rf, file = paste0('data/soil/Model_selection/wheat/', i , '_ML_model_obs.vs.pred_county_mean_rmse.csv'))
  
  rf_county_mean_plot <- test_rf_res.1 %>% 
    ggplot(aes(x = yield_mean, y = pred_county_mean,  color = state_alpha)) + 
    stat_poly_line() +
    stat_poly_eq() +
    geom_abline(color = "gray50", lty = 2) + 
    geom_point(alpha = 0.5) + 
    coord_obs_pred() + 
    labs(x = "observed", y = "predicted") 
  
  rf_county_mean_plot <- rf_county_mean_plot +
    ggplot2::annotate('text', y = max(test_rf_res.1$yield_mean)-0.5, x = min(test_rf_res.1$yield_mean)+0.5, label = paste0('rmse = ',rmse_rf))
  
  ggsave(rf_county_mean_plot, filename = paste0('data/soil/Model_selection/wheat/', i , '_ML_model_obs.vs.pred_county_mean.png'))
  
  
  
  #' 
  #' # Explain the final model using Partical dependence profile: Partial dependence profiles show how the expected value of a model prediction, like the predicted yield, changes as a function of om. See details here: https://ema.drwhy.ai/partialDependenceProfiles.html
  #' 
  ## ----explain---------------------------------------------------------------------------------------------------------------------------------------
  library(DALEXtra)
  
  # create a ggplot function
  ggplot_pdp <- function(obj, x) {
    
    p <- 
      as_tibble(obj$agr_profiles) %>%
      mutate(`_label_` = stringr::str_remove(`_label_`, "^[^_]*_")) %>%
      ggplot(aes(`_x_`, `_yhat_`)) +
      geom_line(data = as_tibble(obj$cp_profiles),
                aes(x = {{ x }}, group = `_ids_`),
                linewidth = 0.5, alpha = 0.5, color = "gray50")
    
    num_colors <- n_distinct(obj$agr_profiles$`_label_`)
    
    if (num_colors > 1) {
      p <- p + geom_line(aes(color = `_label_`), linewidth = 1.2, alpha = 0.8)
    } else {
      p <- p + geom_line(color = "midnightblue", linewidth = 1.2, alpha = 0.8)
    }
    
    p
  }
  
  # pdp ggplot for rf model
  explainer_rf <- 
    explain_tidymodels(
      model_final, 
      data = data, 
      y = wheat_train$Yield_decomp_add,
      label = "cart",
      verbose = FALSE
    )
  
  om_rf <- model_profile(explainer_rf, N = NULL, variables = "ssurgo_om_mean")
  
  # ggplot
  
  pdp_plot_rf <- ggplot_pdp(om_rf, ssurgo_om_mean)  +
    labs(x = "om (%) ", 
         y = "Predicted county mean yield (Mg/ha)", 
         color = NULL) +
    geom_point(data = test_rf_res.1, aes(x = ssurgo_om_mean.y, y = pred_county_mean), color = 'red')  +
    theme_bw()
  
  ggsave(pdp_plot_rf, filename = paste0('data/soil/Model_selection/wheat/',i, '_ML_model_pdp_profile.png'))
  
  
  
  #' 
  #' # Compute all possible yield predictions by changes of SOM from min to max for each county
  #' 
  ## ----predition-------------------------------------------------------------------------------------------------------------------------------------
  min_om <- round(min(test_data$ssurgo_om_mean), digits = 2) #0.37
  max_om <- round(max(test_data$ssurgo_om_mean), digits = 2) #5.89
  
  library(purrr)
  all_om_test <- test_data %>%
    left_join(comet_sum, by = 'GEOID') %>%
    group_by(GEOID, state_alpha) %>%
    nest() %>%
    mutate(grid = map(data, ~c(seq(from = 0.37, to = .$ssurgo_om_mean, 0.2), 
                               seq(from=.$ssurgo_om_mean, to = .$ssurgo_om_mean+.$total_SOM10, 0.01),
                               seq(from=.$ssurgo_om_mean+.$total_SOM10, to =7, 0.5)))) %>%
    unnest(cols = grid) %>%
    select(-data)  %>%
    left_join(test_data, by = c('GEOID','state_alpha')) %>%
    rename(ssurgo_om_county_mean = ssurgo_om_mean,
           ssurgo_om_mean = grid) # rename the grid column to ssurgo_om_mean to be used in prediction
  
  pred_rf_om_grid <- predict(model_final, all_om_test) %>% rename(pred_yield = .pred)
  
  # produce prediction interval and confidence interval
  # check which parsnip supported models have modules for prediction intervals or quantiles#
  #envir <- parsnip::get_model_env()
  #ls(envir) %>% 
  #  tibble(name = .) %>% 
  #  filter(str_detect(name, "_predict")) %>% 
  #  mutate(prediction_modules  = map(name, parsnip::get_from_env)) %>% 
  #  unnest(prediction_modules) %>% 
  #  filter(str_detect(type, "pred_int|quantile"))
  
  # since there is no built-in functions to generate predition intervals for randomforest/boosting trees, 
  #I will use the 'workboots' package to generate bootstrap prediction intervals
  
  library(workboots)
  
  # generate predictions from 2000 bootstrap models
  set.seed(345)
  pred_int_final_wf <-
    model_final %>%
    predict_boots(
      n = 2000,
      training_data = wheat_train,
      new_data = all_om_test
    )
  
  # summarise predictions with a 95% prediction interval
  pred_int_final_wf_om_grid <- pred_int_final_wf %>%
    summarise_predictions() %>%
    select( .pred, .pred_lower, .pred_upper) %>%
    rename(boot_pred = .pred, pred_lower = .pred_lower, pred_upper = .pred_upper)
  
  all_om_test_pred_int <- bind_cols(all_om_test, pred_rf_om_grid, pred_int_final_wf_om_grid)  %>%
    select(GEOID,ssurgo_om_mean,ssurgo_om_county_mean, pred_yield,boot_pred, pred_lower, pred_upper) %>%
    left_join(meta_data, by = c('GEOID','state_alpha'))
  
  write.csv(all_om_test_pred_int, file = paste0('data/soil/Coefficients/wheat/', i , '_ML_model_OM_range_yield_predictions_intervals.csv'))
  
  # generate predictions from 2000 bootstrap models
  #set.seed(345)
  #conf_int_final_wf <-
  #  model_final %>%
  #  predict_boots(
  #    n = 2000,
  #    training_data = wheat_train,
  #    new_data = all_om_test,
  #    interval = 'confidence'
  #  )
  
  # summarise predictions with a 95% confidence interval
  #conf_int_final_wf_om_grid <- conf_int_final_wf %>%
  #  summarise_predictions() %>%
  #  select( .pred, .pred_lower, .pred_upper) %>%
  #  rename(boot_pred = .pred, conf_lower = .pred_lower, conf_upper = .pred_upper)
  
  #all_om_test_conf_int <- bind_cols(all_om_test, pred_rf_om_grid, conf_int_final_wf_om_grid)  %>%
  #  select(GEOID,ssurgo_om_mean,ssurgo_om_county_mean, pred_yield,boot_pred, conf_lower, conf_upper) %>%
  #  left_join(meta_data, by = c('GEOID','state_alpha'))
  
  #write.csv(all_om_test_conf_int, file = paste0('data/soil/Coefficients/wheat/', i , '_ML_model_OM_range_yield_confidence_intervals.csv'))
  
  gc()
  
} # end of the for loop


#' 
#' 
#' # Convert .Rmd to a .R script
#' 

#' 
#' 
#' # Save for machine learning models (not run)
#' 

#' 
