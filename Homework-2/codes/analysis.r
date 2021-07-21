rm(list = ls())
set.seed(1234)
library(keras)
# install_keras()

# Task 1 ------------------------------------------------------------------

# import data
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# a, Show some example images from the data.

# split the canvas into rows and columns
par(mfrow = c(2, 3))

# plot the first 6 images
for (i in 1:6){
  image(
    x_train[i,,],
    col = gray.colors(255), xlab = y_train[i], ylab = ""
  )
}

# b, Train a fully connected deep network to predict items.

# split train data into train and validation sets
# get indexes
set.seed(1234)
train_indices <- sample(seq(nrow(x_train)), 10000)

# split x
data_train_x <- x_train[train_indices,, ]
data_valid_x <- x_train[-train_indices,, ]

# split y
data_train_y <- y_train[train_indices]
data_valid_y <- y_train[-train_indices]

# rescale x
data_train_x <- as.matrix(as.data.frame.array(data_train_x)) / 255
data_valid_x <- as.matrix(as.data.frame.array(data_valid_x)) / 255
data_test_x <- as.matrix(as.data.frame.array(x_test)) / 255

# one-hot encoding for y
data_train_y <- to_categorical(data_train_y, 10)
data_valid_y <- to_categorical(data_valid_y, 10)
data_test_y <- to_categorical(y_test, 10)

### BASELINE MODEL
# create model object - baseline Janos
model_janos <- keras_model_sequential()

# add layers to the model
model_janos %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% # 784 pixels
  layer_dropout(rate = 0.3) %>% # we leave 30% of the node weights unchanged
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model_janos)

# set loss function and the metrics we want to see
compile(
  model_janos,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# janosRDS_NN <- fit(
#   model_janos, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(janosRDS_NN, file = "data/janosRDS_NN.rds")
janosRDS_NN <- readRDS(file = "data/janosRDS_NN.rds")

# Save the model
# save_model_hdf5(model_janos, "data/model_janos.h5")

# Recreate the exact same model purely from the file
model_janos <- load_model_hdf5("data/model_janos.h5")

# evaluate model
model_janos_eval <- as.data.frame(evaluate(model_janos, data_valid_x, data_valid_y))

### FIRST MODEL
# create model object
model1 <- keras_model_sequential()

# add layers to the model
model1 %>%
  layer_dense(units = 28, activation = 'relu', input_shape = c(784)) %>% # 784 pixels
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model1)

# set loss function and the metrics we want to see
compile(
  model1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# model1RDS_NN <- fit(
#   model1, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model1RDS_NN, file = "data/model1RDS_NN.rds")
model1RDS_NN <- readRDS(file = "data/model1RDS_NN.rds")

# Save the model
# save_model_hdf5(model1, "data/model1.h5")

# Recreate the exact same model purely from the file
model1 <- load_model_hdf5("data/model1.h5")

# evaluate model
model1_eval <- as.data.frame(evaluate(model1, data_valid_x, data_valid_y))

### SECOND MODEL
# create model object
model2 <- keras_model_sequential()

# add layers to the model
model2 %>%
  layer_dense(units = 28, activation = 'relu', input_shape = c(784)) %>% # 784 pixels
  layer_dense(units = 28, activation = 'sigmoid') %>%
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model2)

# set loss function and the metrics we want to see
compile(
  model2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# model2RDS_NN <- fit(
#   model2, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model2RDS_NN, file = "data/model2RDS_NN.rds")
model2RDS_NN <- readRDS(file = "data/model2RDS_NN.rds")

# evaluate model
model2_eval <- as.data.frame(evaluate(model2, data_valid_x, data_valid_y))

# Save the model
# save_model_hdf5(model2, "data/model2.h5")

# Recreate the exact same model purely from the file
model2 <- load_model_hdf5("data/model2.h5")


### THIRD MODEL - TOO COMPLEX --> OVERFIT
# create model object
model3 <- keras_model_sequential()

# add layers to the model
model3 %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(784)) %>% # 784 pixels
  layer_dense(units = 100, activation = 'sigmoid') %>%
  layer_dense(units = 100, activation = 'tanh') %>% # tangential function
  layer_dropout(rate = 0.3) %>% # we leave 30% of the node weights unchanged
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model3)

# set loss function and the metrics we want to see
compile(
  model3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# model3RDS_NN <- fit(
#   model3, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model3RDS_NN, file = "data/model3RDS_NN.rds")
model3RDS_NN <- readRDS(file = "data/model3RDS_NN.rds")

# Save the model
# save_model_hdf5(model3, "data/model3.h5")

# Recreate the exact same model purely from the file
model3 <- load_model_hdf5("data/model3.h5")

# evaluate model
model3_eval <- as.data.frame(evaluate(model3, data_valid_x, data_valid_y))

### FOURTH MODEL
# create model object
model4 <- keras_model_sequential()

# add layers to the model
model4 %>%
  layer_dense(units = 100, activation = 'sigmoid', input_shape = c(784)) %>%
  layer_dense(units = 100, activation = 'tanh') %>% # tangential function
  layer_dropout(rate = 0.7) %>% # we leave 70% of the node weights unchanged
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model4)

# set loss function and the metrics we want to see
compile(
  model4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# model4RDS_NN <- fit(
#   model4, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )
# 
# saveRDS(model4RDS_NN, file = "data/model4RDS_NN.rds")
model4RDS_NN <- readRDS(file = "data/model4RDS_NN.rds")

# Save the model
# save_model_hdf5(model4, "data/model4.h5")

# Recreate the exact same model purely from the file
model4 <- load_model_hdf5("data/model4.h5")

# evaluate model
model4_eval <- as.data.frame(evaluate(model4, data_valid_x, data_valid_y))

### FIFTH MODEL --> signs of overfitting
# create model object
model5 <- keras_model_sequential()

# add layers to the model
model5 %>%
  layer_dense(units = 100, activation = 'softplus', input_shape = c(784)) %>% # not really necessary because inputs are not negative
  layer_dense(units = 100, activation = 'relu') %>% # standard bent linear model
  layer_dropout(rate = 0.01) %>% # we leave 70% of the node weights unchanged
  layer_dense(units = 10, activation = 'softmax') # 10 nodes because of the 10 digits

# check model summary
summary(model5)

# set loss function and the metrics we want to see
compile(
  model5,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit model
# set.seed(1234)
# model5RDS_NN <- fit(
#   model5, data_train_x, data_train_y,
#   epochs = 30, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )
# 
# saveRDS(model5RDS_NN, file = "data/model5RDS_NN.rds")
model5RDS_NN <- readRDS(file = "data/model5RDS_NN.rds")

# Save the model
# save_model_hdf5(model5, "data/model5.h5")

# Recreate the exact same model purely from the file
model5 <- load_model_hdf5("data/model5.h5")

# evaluate model
model5_eval <- as.data.frame(evaluate(model5, data_valid_x, data_valid_y))

# model comparison
model_comparison <- cbind(model_janos_eval, model1_eval, model2_eval, model3_eval, model4_eval, model5_eval)

colnames(model_comparison) <- c('baseline', 'model1', 'model2', 'model3', 'model4', 'model5')

knitr::kable(model_comparison, caption = 'NN model comparison', digits = 2) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

plot(model5RDS_NN)


# c, Evaluate the model on the test set. How does test error compare to validation error?

model5_eval_test <- as.data.frame(evaluate(model5, data_test_x, data_test_y))

final_model_comparison <- rbind(as.data.frame(evaluate(model5, data_train_x, data_train_y)) ,model5_eval, model5_eval_test)

rownames(final_model_comparison) <- c('training set', 'validation set', 'test set')

knitr::kable(final_model_comparison, caption = 'Final model comparison across data partitions', digits = 2) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))
# test set performance is expected to be lower as it has fewer observations
# accuracy goes down a little, however loss is more noticeable due to smaller dataset

# d, Try building a convolutional neural network and see if you can improve test set performance.

data_train_x <- array_reshape(data_train_x, c(nrow(data_train_x), 28, 28, 1))
data_valid_x <- array_reshape(data_valid_x, c(nrow(data_valid_x), 28, 28, 1))
data_test_x <- array_reshape(data_test_x, c(nrow(data_test_x), 28, 28, 1))

### JANOS BASELINE MODEL - CNN

cnn_model_janos <- keras_model_sequential()
cnn_model_janos %>%
  layer_conv_2d(
    filters = 32, # number of nodes in this layer (could be anything?!)
    kernel_size = c(3, 3), # search across the input space in 3x3 squares rather than 1x1 squares (which are just individual pixels)
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% # converts space by taking max value from each 2x2 squares
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% # increases complexity
  layer_dense(units = 16, activation = 'relu') %>% # arbitrary
  layer_dense(units = 10, activation = 'softmax') # arbitrary

# Number of parameters:
# - `layer_conv_2d` turns 28 x 28 to 26 x 26, using 9 parameters for each filter (3 x 3 weights), 
# plus a bias for each filter, altogether 320 parameters
# - `max_pooling2d` takes each disjoint 2 x 2 squares and collapses them to 1, turning a 26 x 26
# "image" to a 13 x 13. No parameters are associated with this step.
# - `flatten`: turns each "pixel" in each node to one separate node: 13 x 13 x 32 = 5408
# - `dense`: fully connected layer: 5408 nodes x 16 new nodes + 16 biases = 86544
# - final fully connected layer: 16 x 10 + 10 = 170

compile(
  cnn_model_janos,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# janosRDS_CNN <- fit(
#   cnn_model_janos, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(janosRDS_CNN, file = "data/janosRDS_CNN.rds")
janosRDS_CNN <- readRDS(file = "data/janosRDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model_janos, "data/cnn_model_janos.h5")

# Recreate the exact same model purely from the file
cnn_model_janos <- load_model_hdf5("data/cnn_model_janos.h5")

# evaluate model
model_janos_cnn_eval <- as.data.frame(evaluate(cnn_model_janos, data_valid_x, data_valid_y))

### CNN Model 1 --> less complex version of Janos' model
# lower performance, but train & valid performance is streamlined

cnn_model1 <- keras_model_sequential()
cnn_model1 %>%
  layer_conv_2d(
    filters = 10, # number of nodes in this layer (could be anything?!)
    kernel_size = c(5, 5), # search across the input space in 5x5 squares rather than 1x1 squares (which are just individual pixels)
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% # converts space by taking max value from each 3x3 squares 
  layer_dropout(rate = 0.25) %>%
  # Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. 
  # First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. 
  # MINIST has 10 output classes, so you use a final Dense layer with 10 outputs and a softmax activation.
  layer_flatten() %>% # increases complexity
  layer_dense(units = 10, activation = 'relu') %>% # arbitrary
  layer_dense(units = 10, activation = 'softmax') # arbitrary

compile(
  cnn_model1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# model1RDS_CNN <- fit(
#   cnn_model1, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model1RDS_CNN, file = "data/model1RDS_CNN.rds")
model1RDS_CNN <- readRDS(file = "data/model1RDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model1, "data/cnn_model1.h5")

# Recreate the exact same model purely from the file
cnn_model1 <- load_model_hdf5("data/cnn_model1.h5")

# evaluate model
model1_cnn_eval <- as.data.frame(evaluate(cnn_model1, data_valid_x, data_valid_y))

### CNN Model 2 --> following a stackoverflow example

cnn_model2 <- keras_model_sequential()
cnn_model2 %>%
  layer_conv_2d(
    filters = 10, # number of nodes in this layer (could be anything?!)
    kernel_size = c(5, 5), # search across the input space in 5x5 squares rather than 1x1 squares (which are just individual pixels)
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% # converts space by taking max value from each 3x3 squares
  layer_conv_2d(
    filters = 20, # number of nodes in this layer (could be anything?!)
    kernel_size = c(5, 5), # search across the input space in 5x5 squares rather than 1x1 squares (which are just individual pixels)
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% # converts space by taking max value from each 3x3 squares
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% # increases complexity
  layer_dense(units = 10, activation = 'relu') %>% # arbitrary
  layer_dense(units = 10, activation = 'softmax') # arbitrary

compile(
  cnn_model2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# model2RDS_CNN <- fit(
#   cnn_model2, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model2RDS_CNN, file = "data/model2RDS_CNN.rds")
model2RDS_CNN <- readRDS(file = "data/model2RDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model2, "data/cnn_model2.h5")

# Recreate the exact same model purely from the file
cnn_model2 <- load_model_hdf5("data/cnn_model2.h5")

# evaluate model
model2_cnn_eval <- as.data.frame(evaluate(cnn_model2, data_valid_x, data_valid_y))

### CNN Model 3 --> following a stackoverflow example

cnn_model3 <- keras_model_sequential()
cnn_model3 %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% # increases complexity
  layer_dense(units = 10, activation = 'relu') %>% # arbitrary
  layer_dense(units = 10, activation = 'softmax') # arbitrary

compile(
  cnn_model3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# model3RDS_CNN <- fit(
#   cnn_model3, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model3RDS_CNN, file = "data/model3RDS_CNN.rds")
model3RDS_CNN <- readRDS(file = "data/model3RDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model3, "data/cnn_model3.h5")

# Recreate the exact same model purely from the file
cnn_model3 <- load_model_hdf5("data/cnn_model3.h5")

# evaluate model
model3_cnn_eval <- as.data.frame(evaluate(cnn_model3, data_valid_x, data_valid_y))

### CNN Model 4

cnn_model4 <- keras_model_sequential()
cnn_model4 %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "sigmoid", 
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% # increases complexity
  layer_dense(units = 10, activation = 'softmax') # arbitrary

compile(
  cnn_model4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# model4RDS_CNN <- fit(
#   cnn_model4, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model4RDS_CNN, file = "data/model4RDS_CNN.rds")
model4RDS_CNN <- readRDS(file = "data/model4RDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model4, "data/cnn_model4.h5")

# Recreate the exact same model purely from the file
cnn_model4 <- load_model_hdf5("data/cnn_model4.h5")

# evaluate model
model4_cnn_eval <- as.data.frame(evaluate(cnn_model4, data_valid_x, data_valid_y))

### CNN Model 5

cnn_model5 <- keras_model_sequential()
cnn_model5 %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% # increases complexity
  layer_dense(units = 32, activation = 'relu') %>%  # arbitrary
  layer_dense(units = 32, activation = 'sigmoid') %>%  # arbitrary
  layer_dense(units = 10, activation = 'softmax') # arbitrary

compile(
  cnn_model5,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# set.seed(1234)
# model5RDS_CNN <- fit(
#   cnn_model5, data_train_x, data_train_y,
#   epochs = 15, batch_size = 128,
#   validation_data = list(data_valid_x, data_valid_y)
# )

# saveRDS(model5RDS_CNN, file = "data/model5RDS_CNN.rds")
model5RDS_CNN <- readRDS(file = "data/model5RDS_CNN.rds")

# Save the model
# save_model_hdf5(cnn_model5, "data/cnn_model5.h5")

# Recreate the exact same model purely from the file
cnn_model5 <- load_model_hdf5("data/cnn_model5.h5")

# evaluate model
model5_cnn_eval <- as.data.frame(evaluate(cnn_model5, data_valid_x, data_valid_y))

# model comparison
model_comparison_cnn <- rbind(model_janos_cnn_eval, model1_cnn_eval, model2_cnn_eval, model3_cnn_eval, model4_cnn_eval, model5_cnn_eval)

rownames(model_comparison_cnn) <- c('baseline', 'model1', 'model2', 'model3', 'model4', 'model5')

knitr::kable(model_comparison_cnn, caption = 'CNN model comparison', digits = 2) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

plot(janosRDS_CNN) # plot best model

# Evaluate the model on the test set. How does test error compare to validation error?

model_janos_cnn_eval_test <- as.data.frame(evaluate(cnn_model_janos, data_test_x, data_test_y))

final_model_comparison_cnn <- rbind(as.data.frame(evaluate(cnn_model_janos, data_train_x, data_train_y)) ,model_janos_cnn_eval, model_janos_cnn_eval_test)

rownames(final_model_comparison_cnn) <- c('training set', 'validation set', 'test set')

knitr::kable(final_model_comparison_cnn, caption = 'Final CNN model (baseline - Janos) comparison across data partitions', digits = 2) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))
# test set performance is expected to be lower as it has fewer observations --> not the case
# definitely much better than neural network; convolutional layer makes difference

## analogy for convolutional performance
# - like running PCA pre-processing, getting PCA predictions, 
# and then converting back to level to add level predictions on top of PCA predictions


# Task 2 --------------------------------------------------------------------

library(here)
library(grid)
library(magick)
library(filesstrings)
library(kableExtra)

# a, Pre-process data so that it is acceptable by Keras (set folder structure, bring images to the same size, etc).

# # example photo - trial
# example_image_path <- file.path(here(), "data/hot-dog-not-hot-dog/train/hot_dog/1000288.jpg")
# image_read(example_image_path)  # this is a PIL image
# img <- image_load(example_image_path, target_size = c(150, 150))  # this is a PIL image
# x <- image_to_array(img) / 255
# grid::grid.raster(x)
# 
# xx <- flow_images_from_data(
#   array_reshape(x * 255, c(1, dim(x))),  # take the previous image as base, multiplication is only to conform with the image generator's rescale parameter
#   generator = train_datagen
# )
# augmented_versions <- lapply(1:10, function(ix) generator_next(xx) %>%  {.[1, , , ]})
# # see examples by running in console:
# grid::grid.raster(augmented_versions[[10]])

# set parameters for data partitions
# set augmentation parameters just for the training set
train_datagen <- image_data_generator(rescale = 1/255)  
validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255) 

image_size <- c(150, 150)
batch_size <- 50

set.seed(1234)
validation_indeces_yes <- sample(list.files(paste0(path, "data/hot-dog-not-hot-dog/train/hot_dog")), size = 150, replace = F)
validation_indeces_no <- sample(list.files(paste0(path, "data/hot-dog-not-hot-dog/train/not_hot_dog")), size = 150, replace = F)

dir.create(paste0(path,"data/hot-dog-not-hot-dog/validation"))
dir.create(paste0(path,"data/hot-dog-not-hot-dog/validation/hot_dog"))
dir.create(paste0(path,"data/hot-dog-not-hot-dog/validation/not_hot_dog"))

# move yes cases
for (file in list.files(paste0(path, "data/hot-dog-not-hot-dog/train/hot_dog"))) {
  if (file %in% validation_indeces_yes){
    file.move(paste0(path, "data/hot-dog-not-hot-dog/train/hot_dog/", file), paste0(path,"data/hot-dog-not-hot-dog/validation/hot_dog/"))
  }
}

# move no cases
for (file in list.files(paste0(path, "data/hot-dog-not-hot-dog/train/not_hot_dog"))) {
  if (file %in% validation_indeces_no){
    file.move(paste0(path, "data/hot-dog-not-hot-dog/train/not_hot_dog/", file), paste0(path,"data/hot-dog-not-hot-dog/validation/not_hot_dog/"))
  }
}

train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/validation"),   
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/test"), # Target directory  
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

# b, Estimate a convolutional neural network to predict if an image contains a hot dog or not. Evaluate your model on the test set.

### BASELINE MODEL --> Overfitted
# 3rd dimension refers to the color dimension
hot_dog_model_baseline <- keras_model_sequential()%>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model_baseline %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

# set.seed(1234)
# hot_dog_baseline_RDS <- hot_dog_model_baseline %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 20,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size # divide size of training dataset by batch size
# )
# 
# saveRDS(hot_dog_baseline_RDS, file = "data/hot_dog_baseline_RDS.rds")
hot_dog_baseline_RDS <- readRDS(file = "data/hot_dog_baseline_RDS.rds")

# Save the model
# save_model_hdf5(hot_dog_model_baseline, "data/hot_dog_model_baseline.h5")

# Recreate the exact same model purely from the file
hot_dog_model_baseline <- load_model_hdf5("data/hot_dog_model_baseline.h5")

# evaluate model
hot_dog_model_baseline_eval <- as.data.frame(evaluate(hot_dog_model_baseline, test_generator))
# LIKELY TO BE THE WORST MODEL IN THE HISTORY OF THIS CLASS

plot(hot_dog_baseline_RDS) # --> terrible
knitr::kable(hot_dog_model_baseline_eval, caption = 'Test set performance of baseline CNN image classifier') %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

### FINAL MODEL
hot_dog_model_final <- keras_model_sequential()%>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model_final %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

# set.seed(1234)
# hot_dog_final_RDS <- hot_dog_model_final %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 20,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )
# 
# saveRDS(hot_dog_final_RDS, file = "data/hot_dog_final_RDS.rds")
hot_dog_final_RDS <- readRDS(file = "data/hot_dog_final_RDS.rds")

# Save the model
# save_model_hdf5(hot_dog_model_final, "data/hot_dog_model_final.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final <- load_model_hdf5("data/hot_dog_model_final.h5")

# evaluate model
hot_dog_final_model_eval <- as.data.frame(evaluate(hot_dog_model_final, test_generator))
# better than a coin-flip yo

plot(hot_dog_final_RDS) # --> terrible
knitr::kable(hot_dog_final_model_eval, caption = 'Test set performance of chosen CNN image classifier') %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))


# c, Could data augmentation techniques help with achieving higher predictive accuracy? 

# initialize empty model eval list
model_eval_list <- list()
model_eval_list['final_model_no_augmentation'] <- as.data.frame(evaluate(hot_dog_model_final, validation_generator))

### FIRST DATA AUGMENTATION
# reinitialize training generator for data augmentation
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2, # crops corners of most photos --> useful!
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_final_aug1_RDS <- hot_dog_model_final %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 20,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )
# 
# # Save the model
# save_model_hdf5(hot_dog_model_final, "data/hot_dog_model_aug1.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final <- load_model_hdf5("data/hot_dog_model_aug1.h5")

# saveRDS(hot_dog_final_aug1_RDS, file = "data/hot_dog_final_aug1_RDS.rds")
hot_dog_final_aug1_RDS <- readRDS(file = "data/hot_dog_final_aug1_RDS.rds")

model_eval_list['first_try_augmentation'] <- as.data.frame(evaluate(hot_dog_model_final, validation_generator))


### SECOND DATA AUGMENTATION
# reinitialize training generator for data augmentation
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 90,
  width_shift_range = 0.4,
  height_shift_range = 0.4,
  zoom_range = 0.5, # crops corners of most photos --> useful!
  brightness_range = c(1,1.5),
  fill_mode = "nearest" # applies to new sections of image after rotations etc.
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_final_aug2_RDS <- hot_dog_model_final %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 20,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )
# 
# # Save the model
# save_model_hdf5(hot_dog_model_final, "data/hot_dog_model_aug2.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final <- load_model_hdf5("data/hot_dog_model_aug2.h5")

# saveRDS(hot_dog_final_aug2_RDS, file = "data/hot_dog_final_aug2_RDS.rds")
hot_dog_final_aug2_RDS <- readRDS(file = "data/hot_dog_final_aug2_RDS.rds")

model_eval_list['second_try_augmentation'] <- as.data.frame(evaluate(hot_dog_model_final, validation_generator))
# accuracy is significantly higher --> proof that augmentation on training set works?!


### THIRD DATA AUGMENTATION
# reinitialize training generator for data augmentation
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 10,
  width_shift_range = 0.5,
  height_shift_range = 0.5,
  zoom_range = 0.1, # crops corners of most photos --> useful!
  brightness_range = c(1.5,2),
  fill_mode = "wrap" # applies to new sections of image after rotations etc.
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_final_aug3_RDS <- hot_dog_model_final %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 20,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )

# Save the model
# save_model_hdf5(hot_dog_model_final, "data/hot_dog_model_aug3.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final <- load_model_hdf5("data/hot_dog_model_aug3.h5")

# saveRDS(hot_dog_final_aug3_RDS, file = "data/hot_dog_final_aug3_RDS.rds")
hot_dog_final_aug3_RDS <- readRDS(file = "data/hot_dog_final_aug3_RDS.rds")

model_eval_list['third_try_augmentation'] <- as.data.frame(evaluate(hot_dog_model_final, validation_generator))
# model performance seems to stay relatively constant between training and validation

model_eval_hot_dog <- as.data.frame(model_eval_list)
row.names(model_eval_hot_dog) <- c('loss','accuracy')
knitr::kable(model_eval_hot_dog, caption = "", digits = 3 ) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

## Conclusion to this point - augmentation does help performance but we cannot choose best model as this is on test set

# d, Try to rely on some pre-built neural networks to aid prediction. 
# Can you achieve a better performance using transfer learning for this problem?

##### FIRST TRY
# we will improve the augmented final model
# create the base pre-trained model
conv_base <- application_inception_v3(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(image_size, 3)
)

summary(conv_base)
# The convolutional base of inception has 21,768,352 parameters, which is very large

# freeze all convolutional inception V3 layers
freeze_weights(conv_base)

### FINAL MODEL + Transfer Learning w/ Inception V3
hot_dog_model_final_inceptionV3 <- keras_model_sequential()%>%
  conv_base %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

hot_dog_model_final_inceptionV3 %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

# reinitialize training generator for data augmentation with best augmentation parameters
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2, # crops corners of most photos --> useful!
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_model_final_inceptionV3_RDS <- hot_dog_model_final_inceptionV3 %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 10,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )
# 
# # Save the model
# save_model_hdf5(hot_dog_model_final_inceptionV3, "data/hot_dog_model_final_inceptionV3.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final_inceptionV3 <- load_model_hdf5("data/hot_dog_model_final_inceptionV3.h5")

# saveRDS(hot_dog_model_final_inceptionV3_RDS, file = "data/hot_dog_model_final_inceptionV3_RDS.rds")
hot_dog_model_final_inceptionV3_RDS <- readRDS(file = "data/hot_dog_model_final_inceptionV3_RDS.rds")

model_eval_list['final_conv_inceptionV3_augmented'] <- 
  as.data.frame(evaluate(hot_dog_model_final_inceptionV3, validation_generator))

##### SECOND TRY
# we will improve the augmented final model
# create the base pre-trained model
base_model <- application_mobilenet(
  weights = 'imagenet', 
  include_top = FALSE,
  input_shape = c(image_size, 3))

summary(base_model)
# The convolutional base of mobilenet has 3,228,864 parameters, which is also very large

# freeze all convolutional mobilenet layers
freeze_weights(base_model)

### FINAL MODEL + Transfer Learning w/ Mobilenet
hot_dog_model_final_mobilenet <- keras_model_sequential()%>%
  base_model %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

hot_dog_model_final_mobilenet %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

# reinitialize training generator for data augmentation with best augmentation parameters
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2, # crops corners of most photos --> useful!
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_model_final_mobilenet_RDS <- hot_dog_model_final_mobilenet %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 10,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )

# Save the model
# save_model_hdf5(hot_dog_model_final_mobilenet, "data/hot_dog_model_final_mobilenet.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final_mobilenet <- load_model_hdf5("data/hot_dog_model_final_mobilenet.h5")

# saveRDS(hot_dog_model_final_mobilenet_RDS, file = "data/hot_dog_model_final_mobilenet_RDS.rds")
hot_dog_model_final_mobilenet_RDS <- readRDS(file = "data/hot_dog_model_final_mobilenet_RDS.rds")

model_eval_list['final_conv_mobilenet_augmented'] <- 
  as.data.frame(evaluate(hot_dog_model_final_mobilenet, validation_generator))

##### THIRD TRY
# we will improve the augmented final model
# create the base pre-trained model
resnet50_model <- application_resnet50(
  input_shape = c(image_size, 3), 
  weights = 'imagenet', 
  include_top = FALSE
)

summary(resnet50_model)
# The convolutional base of resnet has 23,587,712 parameters, which is also very large

# freeze all convolutional resnet layers
freeze_weights(resnet50_model)

### FINAL MODEL + Transfer Learning w/ Resnet
hot_dog_model_final_resnet <- keras_model_sequential()%>%
  resnet50_model %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

hot_dog_model_final_resnet %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

# reinitialize training generator for data augmentation with best augmentation parameters
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2, # crops corners of most photos --> useful!
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)
# set.seed(1234)
# hot_dog_model_final_resnet_RDS <- hot_dog_model_final_resnet %>% fit(
#   train_generator,
#   steps_per_epoch = 198 / batch_size, # divide size of training dataset by batch size
#   epochs = 10,
#   validation_data = validation_generator,
#   validation_steps = 300 / batch_size, # divide size of training dataset by batch size
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1) # if crossentropy loss plateaus, change LR
# )

# Save the model
# save_model_hdf5(hot_dog_model_final_resnet, "data/hot_dog_model_final_resnet.h5")

# Recreate the exact same model purely from the file
hot_dog_model_final_resnet <- load_model_hdf5("data/hot_dog_model_final_resnet.h5")

# saveRDS(hot_dog_model_final_resnet_RDS, file = "data/hot_dog_model_final_resnet_RDS.rds")
hot_dog_model_final_resnet_RDS <- readRDS(file = "data/hot_dog_model_final_resnet_RDS.rds")

model_eval_list['final_conv_resnet_augmented'] <- 
  as.data.frame(evaluate(hot_dog_model_final_resnet, validation_generator))

# take final look at your baby --> all variants of final model
model_eval_hot_dog <- as.data.frame(model_eval_list)
row.names(model_eval_hot_dog) <- c('loss','accuracy')
knitr::kable(model_eval_hot_dog, caption = "", digits = 3 ) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))


