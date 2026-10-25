# ============================================================
# PREDICTIVE ANALYTICS MINOR PROJECT
# Heart Disease Prediction using R
# Dataset: UCI Machine Learning Repository - Heart Disease
# https://archive.ics.uci.edu/dataset/45/heart+disease
# DOI: https://doi.org/10.24432/C52P4X
#
# Imputation, dummy encoding, zero-variance filtering, and scaling
# are fitted inside caret resampling folds using recipes.
# ============================================================

# ------------------------- 1. Packages -------------------------
required <- c(
  "tidyverse", "caret", "rpart", "rpart.plot", "randomForest",
  "e1071", "pROC", "janitor", "recipes", "kernlab"
)

missing_pkgs <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(
    missing_pkgs,
    repos = "https://cloud.r-project.org",
    dependencies = TRUE
  )
}

still_missing <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(still_missing) > 0) {
  stop(
    "Required packages could not be installed: ",
    paste(still_missing, collapse = ", "),
    ". Check the internet connection and installation messages, then rerun Section 1."
  )
}

suppressPackageStartupMessages(
  invisible(lapply(required, library, character.only = TRUE))
)
message("All required packages are installed and loaded.")
message("recipes version: ", as.character(packageVersion("recipes")))
message("kernlab version: ", as.character(packageVersion("kernlab")))

set.seed(2026)
dir.create("project_outputs", showWarnings = FALSE, recursive = TRUE)


# ------------------------- 2. Load data -------------------------
data_url <- paste0(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/",
  "heart-disease/processed.cleveland.data"
)

column_names <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
)

heart_raw <- tryCatch(
  read.csv(
    data_url,
    header = FALSE,
    sep = ",",
    col.names = column_names,
    stringsAsFactors = FALSE,
    na.strings = c("?", "", "NA"),
    strip.white = TRUE
  ),
  error = function(e) {
    stop(
      "Could not download the UCI dataset. Check your internet connection. ",
      "Download processed.cleveland.data from ",
      "https://archive.ics.uci.edu/dataset/45/heart+disease and load it locally. ",
      "Original error: ", conditionMessage(e)
    )
  }
)

heart <- janitor::clean_names(heart_raw)
if (!all(column_names %in% names(heart))) {
  stop("Dataset columns do not match the expected UCI Cleveland file format.")
}
if (nrow(heart) == 0L) stop("The loaded dataset contains no rows.")

target_raw <- suppressWarnings(as.integer(as.character(heart$num)))
if (anyNA(target_raw)) {
  stop("The target column contains missing or non-integer diagnosis values.")
}
if (!all(target_raw %in% 0:4)) {
  stop("Unexpected target values found. Expected original diagnosis values 0 through 4.")
}

heart <- dplyr::rename(heart, target_raw = num)
heart$target_raw <- target_raw

cat("\n--- Dataset dimensions ---\n")
print(dim(heart))
cat("\n--- Column names ---\n")
print(names(heart))
cat("\n--- First six rows ---\n")
print(head(heart))
write.csv(heart, "project_outputs/01_loaded_data.csv", row.names = FALSE)


# ------------------------- 3. Data audit and preparation -------------------------
cat("\n--- Structure ---\n")
str(heart)
cat("\n--- Summary of original dataset ---\n")
print(summary(heart))

missing_summary <- data.frame(
  variable = names(heart),
  missing_count = vapply(heart, function(x) as.numeric(sum(is.na(x))), numeric(1)),
  missing_percent = round(
    vapply(heart, function(x) mean(is.na(x)) * 100, numeric(1)), 2
  ),
  row.names = NULL
)
print(missing_summary)
write.csv(
  missing_summary,
  "project_outputs/02_missing_value_audit.csv",
  row.names = FALSE
)

duplicate_count <- sum(duplicated(heart))
cat("\nDuplicate rows detected:", duplicate_count, "\n")
if (duplicate_count > 0) {
  heart <- heart[!duplicated(heart), , drop = FALSE]
  cat("Duplicate rows removed:", duplicate_count, "\n")
}

# Binary target: 0 = no recorded disease; 1-4 = recorded disease.
heart$target <- ifelse(heart$target_raw == 0, "No_Disease", "Disease")
heart$target <- factor(heart$target, levels = c("Disease", "No_Disease"))
heart$target_raw <- NULL

# Convert coded categorical predictors to factors.
categorical_candidates <- intersect(
  c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"),
  names(heart)
)
for (v in categorical_candidates) heart[[v]] <- as.factor(heart[[v]])

if (anyNA(heart$target)) stop("The binary target contains missing values.")
if (length(unique(heart$target)) != 2L) {
  stop("The target does not contain both classes after transformation.")
}
if (any(!is.finite(heart$age))) stop("Non-finite values detected in age.")

cat("\n--- Binary target distribution ---\n")
print(table(heart$target))
print(round(prop.table(table(heart$target)) * 100, 2))

# Predictor missing values are intentionally not imputed here.
# Recipe-based imputation is fitted within resampling folds below.
write.csv(
  heart, "project_outputs/03_pre_model_data.csv",
  row.names = FALSE, na = ""
)


# ------------------------- 4. Exploratory Data Analysis -------------------------
p1 <- ggplot(heart, aes(x = target, fill = target)) +
  geom_bar() +
  labs(
    title = "Heart Disease Class Distribution",
    x = "Heart Disease Status", y = "Number of Patients"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
print(p1)
ggsave("project_outputs/fig1_target_distribution.png", p1,
       width = 8, height = 5, dpi = 300)

p2 <- ggplot(heart, aes(x = age, fill = target)) +
  geom_histogram(binwidth = 5, position = "identity",
                 alpha = 0.60, boundary = 0) +
  labs(
    title = "Age Distribution by Heart Disease Status",
    x = "Age (years)", y = "Number of Patients", fill = "Diagnosis"
  ) +
  theme_minimal()
print(p2)
ggsave("project_outputs/fig2_age_distribution.png", p2,
       width = 8, height = 5, dpi = 300)

p3 <- ggplot(heart, aes(x = age, y = thalach, color = target)) +
  geom_point(size = 2.2, alpha = 0.75) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Age vs Maximum Heart Rate",
    x = "Age (years)", y = "Maximum Heart Rate", color = "Diagnosis"
  ) +
  theme_minimal()
print(p3)
ggsave("project_outputs/fig3_age_vs_max_heart_rate.png", p3,
       width = 8, height = 5, dpi = 300)

p4 <- ggplot(heart, aes(x = target, y = chol, fill = target)) +
  geom_boxplot(alpha = 0.75) +
  labs(
    title = "Cholesterol Distribution by Diagnosis",
    x = "Heart Disease Status", y = "Serum Cholesterol"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
print(p4)
ggsave("project_outputs/fig4_cholesterol_boxplot.png", p4,
       width = 8, height = 5, dpi = 300)

numeric_predictors <- names(heart)[
  vapply(heart, is.numeric, logical(1))
]
numeric_predictors <- setdiff(numeric_predictors, "target")

if (length(numeric_predictors) > 0) {
  numeric_df <- heart[, numeric_predictors, drop = FALSE]
  desc <- data.frame(
    variable = names(numeric_df),
    mean = vapply(numeric_df, function(x) mean(x, na.rm = TRUE), numeric(1)),
    sd = vapply(numeric_df, function(x) sd(x, na.rm = TRUE), numeric(1)),
    median = vapply(numeric_df, function(x) median(x, na.rm = TRUE), numeric(1)),
    min = vapply(numeric_df, function(x) min(x, na.rm = TRUE), numeric(1)),
    max = vapply(numeric_df, function(x) max(x, na.rm = TRUE), numeric(1)),
    row.names = NULL
  )
  print(desc)
  write.csv(desc, "project_outputs/05_descriptive_statistics.csv", row.names = FALSE)
}

if (length(numeric_predictors) >= 2) {
  corr <- cor(
    heart[, numeric_predictors, drop = FALSE],
    use = "pairwise.complete.obs"
  )
  write.csv(corr, "project_outputs/04_numeric_correlations.csv")
  if (requireNamespace("corrplot", quietly = TRUE)) {
    png("project_outputs/fig5_correlation_matrix.png",
        width = 1200, height = 1000, res = 150)
    corrplot::corrplot(
      corr, method = "color", type = "upper",
      tl.col = "black", tl.cex = 0.8
    )
    dev.off()
  } else {
    message("Optional package 'corrplot' is not installed; correlation CSV saved, plot skipped.")
  }
}
cat("\nEDA visualizations and descriptive statistics have been saved.\n")


# ------------------------- 5. Train/test split -------------------------
set.seed(2026)
train_idx <- caret::createDataPartition(
  y = heart$target, p = 0.80, list = FALSE
)
train_data <- heart[train_idx, , drop = FALSE]
test_data <- heart[-train_idx, , drop = FALSE]

# Drop factor levels absent from the training partition so the recipe can
# recognize any such levels encountered in validation/test data as novel.
train_data <- droplevels(train_data)

train_data$target <- factor(train_data$target,
                            levels = c("Disease", "No_Disease"))
test_data$target <- factor(test_data$target,
                           levels = c("Disease", "No_Disease"))

cat("\nTraining rows:", nrow(train_data), "\n")
cat("Test rows:", nrow(test_data), "\n")
cat("\nTraining target distribution:\n")
print(table(train_data$target))
cat("\nTest target distribution:\n")
print(table(test_data$target))

# Recipe preprocessing is estimated within each resampling fold.
# Missing numeric predictors: median; missing factor predictors: mode.
# Then handle novel categories, dummy-encode, remove zero-variance
# predictors, and normalize numeric columns.
preprocess_recipe <- recipes::recipe(target ~ ., data = train_data) |>
  recipes::step_impute_median(recipes::all_numeric_predictors()) |>
  recipes::step_impute_mode(recipes::all_nominal_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors(), new_level = "new") |>
  recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = FALSE) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_normalize(recipes::all_numeric_predictors())

ctrl <- caret::trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final",
  allowParallel = FALSE
)


# ------------------------- 6. Model development -------------------------
set.seed(2026)
model_glm <- caret::train(
  preprocess_recipe, data = train_data, method = "glm",
  family = binomial(), metric = "ROC", trControl = ctrl
)

set.seed(2026)
model_tree <- caret::train(
  preprocess_recipe, data = train_data, method = "rpart",
  metric = "ROC", trControl = ctrl, tuneLength = 10
)

set.seed(2026)
model_rf <- caret::train(
  preprocess_recipe, data = train_data, method = "rf",
  metric = "ROC", trControl = ctrl, tuneLength = 5,
  ntree = 500, importance = TRUE
)

set.seed(2026)
model_svm <- caret::train(
  preprocess_recipe, data = train_data, method = "svmRadial",
  metric = "ROC", trControl = ctrl, tuneLength = 5
)

models <- list(
  Logistic_Regression = model_glm,
  Decision_Tree = model_tree,
  Random_Forest = model_rf,
  SVM_Radial = model_svm
)

cv_results <- dplyr::bind_rows(lapply(names(models), function(model_name) {
  current <- models[[model_name]]$results
  current$model <- model_name
  current
}))
write.csv(cv_results, "project_outputs/06_cross_validation_results.csv",
          row.names = FALSE)


# ------------------------- 7. Test-set evaluation -------------------------
calc_metrics <- function(model, test_set, model_name) {
  probability_table <- predict(model, newdata = test_set, type = "prob")
  if (!("Disease" %in% names(probability_table))) {
    stop("The model did not return a 'Disease' probability column: ", model_name)
  }

  probs <- probability_table[, "Disease"]
  predicted <- factor(
    ifelse(probs >= 0.50, "Disease", "No_Disease"),
    levels = c("Disease", "No_Disease")
  )
  truth <- factor(test_set$target, levels = c("Disease", "No_Disease"))

  cm <- caret::confusionMatrix(
    data = predicted, reference = truth, positive = "Disease"
  )

  roc_obj <- pROC::roc(
    response = truth,
    predictor = probs,
    levels = c("No_Disease", "Disease"),
    direction = "<",
    quiet = TRUE
  )

  sensitivity <- unname(cm$byClass["Sensitivity"])
  specificity <- unname(cm$byClass["Specificity"])
  precision <- unname(cm$byClass["Pos Pred Value"])

  if (is.na(precision) || is.na(sensitivity) ||
      (precision + sensitivity) == 0) {
    f1 <- NA_real_
  } else {
    f1 <- 2 * precision * sensitivity / (precision + sensitivity)
  }

  metrics <- data.frame(
    model = model_name,
    accuracy = unname(cm$overall["Accuracy"]),
    precision = precision,
    recall_sensitivity = sensitivity,
    specificity = specificity,
    f1_score = f1,
    roc_auc = as.numeric(pROC::auc(roc_obj)),
    row.names = NULL
  )

  write.csv(
    as.data.frame(cm$table),
    paste0("project_outputs/confusion_matrix_", model_name, ".csv"),
    row.names = FALSE
  )

  list(
    metrics = metrics,
    roc = roc_obj,
    probabilities = probs,
    predictions = predicted,
    confusion_matrix = cm
  )
}

evaluation <- lapply(
  names(models),
  function(model_name) calc_metrics(models[[model_name]], test_data, model_name)
)
names(evaluation) <- names(models)

results <- dplyr::bind_rows(lapply(evaluation, function(x) x$metrics))
results <- dplyr::arrange(results, dplyr::desc(roc_auc))

cat("\n--- Held-out test-set model comparison ---\n")
print(results)
write.csv(results, "project_outputs/07_test_set_model_comparison.csv",
          row.names = FALSE)


# ------------------------- 8. ROC curves -------------------------
png("project_outputs/fig6_roc_curves.png",
    width = 1200, height = 900, res = 150)
plot(evaluation[[1]]$roc, main = "ROC Curves on Held-Out Test Set")
if (length(evaluation) > 1) {
  for (i in 2:length(evaluation)) plot(evaluation[[i]]$roc, add = TRUE)
}
auc_values <- vapply(
  evaluation, function(x) as.numeric(pROC::auc(x$roc)), numeric(1)
)
legend(
  "bottomright",
  legend = paste0(
    names(evaluation), " (AUC = ",
    format(round(auc_values, 3), nsmall = 3), ")"
  ),
  lty = 1, cex = 0.75
)
dev.off()


# ------------------------- 9. Random Forest feature importance -------------------------
importance_result <- tryCatch(
  caret::varImp(model_rf, scale = TRUE),
  error = function(e) {
    message("Feature importance could not be computed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(importance_result) && !is.null(importance_result$importance)) {
  importance_df <- data.frame(
    feature = rownames(importance_result$importance),
    importance = rowMeans(importance_result$importance, na.rm = TRUE),
    row.names = NULL
  )
  importance_df <- importance_df[
    order(importance_df$importance, decreasing = TRUE), , drop = FALSE
  ]
  write.csv(importance_df,
            "project_outputs/08_random_forest_feature_importance.csv",
            row.names = FALSE)

  p_imp <- ggplot(
    head(importance_df, 12),
    aes(x = reorder(feature, importance), y = importance)
  ) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Random Forest Feature Importance",
      x = "Feature", y = "Mean scaled importance"
    ) +
    theme_minimal()
  print(p_imp)
  ggsave("project_outputs/fig7_feature_importance.png", p_imp,
         width = 8, height = 5, dpi = 300)
} else {
  message("Feature-importance output was skipped.")
}


# ------------------------- 10. Save model and session information -------------------------
best_name <- results$model[1]
best_model <- models[[best_name]]
saveRDS(best_model, "project_outputs/best_model.rds")
capture.output(sessionInfo(), file = "project_outputs/session_info.txt")

cat("\nAnalysis complete.\n")
cat("Best model by held-out ROC-AUC:", best_name, "\n")
cat("Generated tables, plots, and model files are in project_outputs/.\n")
