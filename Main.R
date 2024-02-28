library(dplyr)
library(stringr)
library(ggplot2)
library(corrplot)
library(randomForest)
library(caret)
library(e1071)
library(rpart)
library(naivebayes)
library(gridExtra)
library(shiny)
library(plotly)
library(wordcloud2)
library(jiebaR)
library(pROC)

## 设置路径
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

## 数据读取&处理
data <- read.csv("HOTELS.csv")
data <- data[, -c(1:10, 51)]

## 切分出star, score, facilities和location
data_star <- data$star
data_star <- as.numeric(gsub("^(\\d(\\.\\d)?) stars out of 5$", "\\1", data_star))

data_score <- data[, 5:11]

## 性价比和cleanliness, facilities, and service比较相关
data_facilities <- data[, c(12:21, 30:40)]
data_locations <- data[, c(22:29)]

## 对facilities矩阵做独热编码处理
one_hot_encode <- function(col) {
  all_elements <- c()
  unique_elements <- c()
  encoded_matrix <- NULL
  
  for (item in col) {
    if (!is.na(item) && item != "") {
      split_elements <- strsplit(item, " @@ ")[[1]]
      all_elements <- c(all_elements, split_elements)
    }
  }
  
  unique_elements <- unique(all_elements)
  n <- length(unique_elements)
  
  encode_item <- function(item) {
    encoded <- rep(0, n)
    if (!is.na(item) && item != "") {
      split_elements <- strsplit(item, " @@ ")[[1]]
      indices <- match(split_elements, unique_elements)
      encoded[indices] <- 1
    }
    return(encoded)
  }
  
  encoded_matrix <- t(sapply(col, encode_item))
  rownames(encoded_matrix) <- 1:nrow(encoded_matrix)
  
  return(list(unique_elements = unique_elements, encoded_matrix = encoded_matrix))
}

facilities_vec <- c("Languages.spoken", "Things.to.do..ways.to.relax", "Cleanliness.and.safety",
                    "Dining..drinking..and.snacking", "Services.and.conveniences", "Access",
                    "Available.in.all.rooms")

facilities_encode <- matrix(0, nrow = 1388, ncol = 0)
for (item in facilities_vec) {
  print(item)
  facilities_encode <- cbind(facilities_encode, one_hot_encode(data_facilities[, item])$encoded_matrix)
}

## 进行PCA/MCA降维
pca_result <- prcomp(facilities_encode, scale. = TRUE)
pca_components <- pca_result$x[, 1:100]

####################
## 对facilities矩阵采用TF-IDF方法，发现效果不是很好
####################
TFIDF <- function(encoded_matrix) {
  TFIDF <- c()
  
  for (i in 1:nrow(encoded_matrix)) {
    TF <- c()
    IDF <- c()
    idx <- which(encoded_matrix[i, ] == 1)
    
    if (length(idx) == 0) {
      TFIDF <- c(TFIDF, 0)
    } else {
      TF <- rep(1, times = length(idx)) / length(idx)
      tmp <- rep(nrow(encoded_matrix), times = length(idx)) / sapply(idx, function(j) sum(encoded_matrix[, j]))
      IDF <- log(tmp)
      TFIDF <- c(TFIDF, TF%*%IDF)
    }
  }
  
  return(TFIDF)
}

facilities_TFIDF <- matrix(0, nrow = 1388, ncol = 0)
for (item in facilities_vec) {
  print(item)
  facilities_TFIDF <- cbind(facilities_TFIDF, TFIDF(one_hot_encode(data_facilities[, item])$encoded_matrix))
}
####################

## 对locations矩阵做处理,提取距离信息
extract_locations <- function(col) {
  extracted_locations <- c()
  
  for (item in col) {
    if (!is.na(item) && item != "") {
      extracted_text <- str_extract(item, "(?<=## ).*?(?<=m)")

      if (grepl("km", extracted_text)) {
        num <- 1000 * as.numeric(unlist(str_extract(extracted_text, "[0-9.]+")))
      } else {
        num <- as.numeric(unlist(str_extract(extracted_text, "[0-9.]+")))
      }
      
    } else {
      num <- NA
    }
    extracted_locations <- c(extracted_locations, num)
  }
  
  return(extracted_locations)
}

locations_vec <- c("Airports", "Public.transportation", "Hospital.or.clinic",
                   "Shopping", "Convenience.store", "Cash.withdrawal", "Popular.landmarks",
                   "Nearby.landmarks")

locations_encode <- matrix(0, nrow = 1388, ncol = 0)
for (item in locations_vec) {
  print(item)
  extract_locations(data_locations[, item])
  locations_encode <- cbind(locations_encode, extract_locations(data_locations[, item]))
}
for (i in 1:ncol(locations_encode)) {
  col_max <- max(locations_encode[, i], na.rm = TRUE)  # 计算每列的最大值
  locations_encode[is.na(locations_encode[, i]), i] <- col_max  # 用最大值填补缺失值
}

## 设置种子,便于复现
set.seed(20231227)
cleaned_data <- as.data.frame(na.omit(cbind(data$price, pca_components, locations_encode)))
idx <- sample(1:nrow(cleaned_data), 0.8 * nrow(cleaned_data))
train_data <- cleaned_data[idx, ]
test_data <- cleaned_data[-idx, -1]
test_label <- cleaned_data[-idx, 1]

## 用logistic regression, support vector machine, decision tree
## naive bayes和random forest做二分类, 以$140~￥1000(中位数)作为分界
cls_lr <- glm(as.factor(ifelse(V1 <= 140, 0, 1)) ~ ., data = train_data, family = binomial)
cls_svm <- svm(as.factor(ifelse(V1 <= 140, 0, 1)) ~ ., data = train_data, kernel = "radial", probability = TRUE)
cls_tree <- rpart(as.factor(ifelse(V1 <= 140, 0, 1)) ~ ., data = train_data, method = "class")
cls_nb <- naive_bayes(as.factor(ifelse(V1 <= 140, 0, 1)) ~ ., data = train_data)
cls_rf <- randomForest(as.factor(ifelse(V1 <= 140, 0, 1)) ~ ., data = train_data)

## 用linear regression, support vector regression, random forest regression做回归
reg_lr <- lm(log(V1) ~ ., data = train_data)
reg_svr <- svm(V1 ~ ., data = train_data, kernel = "radial")
reg_rf <- randomForest(V1 ~ ., data = train_data)

price_lr <- exp(predict(reg_lr, test_data))
price_svr <- predict(reg_svr, test_data)
price_rf <- predict(reg_rf, test_data)

cor(price_lr, test_label)
cor(price_svr, test_label)
cor(price_rf, test_label)

lr_plot <- ggplot(data = data.frame(Price_LR = price_lr, Test_Label = test_label), aes(x = Price_LR, y = Test_Label)) +
  geom_point(color = "black", alpha = 0.6) +  # 设置点的颜色为黑色，设置点的透明度为0.6
  geom_abline(intercept = 0, slope = 1, color = "gray") +  # 添加y=x的灰色直线
  xlim(0, 1000) + ylim(0, 1000) +  # 限制x和y轴的范围在0到1000之间
  labs(x = "Predicted Price ($)", y = "Actual Price ($)") +
  ggtitle("Linear Regression Model") +
  geom_text(
    aes(x = 800, y = 200, label = paste("Corr=", round(cor(price_lr, test_label), 2))),
    size = 4,
    color = "red"
  ) +
  theme_minimal()

# 创建支持向量回归模型的散点图
svr_plot <- ggplot(data = data.frame(Price_SVR = price_svr, Test_Label = test_label), aes(x = Price_SVR, y = Test_Label)) +
  geom_point(color = "black", alpha = 0.6) +  # 设置点的颜色为黑色，设置点的透明度为0.6
  geom_abline(intercept = 0, slope = 1, color = "gray") +  # 添加y=x的灰色直线
  xlim(0, 1000) + ylim(0, 1000) +  # 限制x和y轴的范围在0到1000之间
  labs(x = "Predicted Price ($)", y = "Actual Price ($)") +
  ggtitle("Support Vector Regression Model") +
  geom_text(
    aes(x = 800, y = 200, label = paste("Corr=", round(cor(price_svr, test_label), 2))),
    size = 4,
    color = "red"
  ) +
  theme_minimal()

# 创建随机森林回归模型的散点图
rf_plot <- ggplot(data = data.frame(Price_RF = price_rf, Test_Label = test_label), aes(x = Price_RF, y = Test_Label)) +
  geom_point(color = "black", alpha = 0.6) +  # 设置点的颜色为黑色，设置点的透明度为0.6
  geom_abline(intercept = 0, slope = 1, color = "gray") +  # 添加y=x的灰色直线
  xlim(0, 1000) + ylim(0, 1000) +  # 限制x和y轴的范围在0到1000之间
  labs(x = "Predicted Price ($)", y = "Actual Price ($)") +
  ggtitle("Random Forest Regression Model") +
  geom_text(
    aes(x = 800, y = 200, label = paste("Corr=", round(cor(price_rf, test_label), 2))),
    size = 4,
    color = "red"
  ) +
  theme_minimal()

# 分别显示三张图
print(lr_plot)
print(svr_plot)
print(rf_plot)

price_true <- ifelse(test_label <= 140, 0, 1)
price_pred_lr <- ifelse(predict(cls_lr, test_data, type = "response") >= 0.5, 1, 0)
price_pred_svm <- predict(cls_svm, test_data)
price_pred_tree <- predict(cls_tree, test_data, type = "class")
price_pred_nb <- predict(cls_nb, test_data)
price_pred_rf <- predict(cls_rf, test_data)
price_ensemble <- ifelse(rowMeans(cbind(as.numeric(price_pred_lr),
                                        as.numeric(price_pred_svm) - 1,
                                        as.numeric(price_pred_tree) - 1,
                                        as.numeric(price_pred_nb) - 1,
                                        as.numeric(price_pred_rf) - 1)) >= 0.5, 1, 0)

# 生成ROC曲线
prob_lr <- predict(cls_lr, test_data, type = "response")
prob_svm <- attr(predict(cls_svm, test_data, probability = TRUE), "probabilities")[, 2]
prob_tree <- predict(cls_tree, test_data, type = "prob")[, 2]
prob_nb <- predict(cls_nb, test_data, type = "prob")[, 2]
prob_rf <- predict(cls_rf, test_data, type = "prob")[, 2]
prob_ensemble <- rowMeans(cbind(prob_lr, prob_svm, prob_tree, prob_nb, prob_rf))

# 计算ROC曲线
roc_lr <- roc(price_true, prob_lr)
roc_svm <- roc(price_true, prob_svm)
roc_tree <- roc(price_true, prob_tree)
roc_nb <- roc(price_true, prob_nb)
roc_rf <- roc(price_true, prob_rf)
roc_ensemble <- roc(price_true, prob_ensemble)

# 创建ROC曲线图
roc_plot <- ggroc(
  list(
    "Logistic Regression" = roc_lr,
    "Support Vector Machine" = roc_svm,
    "Decision Tree" = roc_tree,
    "Naive Bayes" = roc_nb,
    "Random Forest" = roc_rf,
    "Ensemble" = roc_ensemble
  ),
  legacy.axes = TRUE
) +
  geom_line(size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme(legend.title = element_blank()) +
  theme_minimal(base_size = 14) +
  # 添加AUC信息标签，并手动指定颜色
  geom_text(
    aes(x = 0.8, y = 0.3, label = paste("AUC =", round(auc(roc_lr), 2))),
    size = 4,
    color = "#ED76AC"  # R237 G118 U108
  ) +
  geom_text(
    aes(x = 0.8, y = 0.25, label = paste("AUC =", round(auc(roc_svm), 2))),
    size = 4,
    color = "#B39E00"  # R179 G158 U0
  ) +
  geom_text(
    aes(x = 0.8, y = 0.2, label = paste("AUC =", round(auc(roc_tree), 2))),
    size = 4,
    color = "#3DB933"  # R61 G185 U51
  ) +
  geom_text(
    aes(x = 0.8, y = 0.15, label = paste("AUC =", round(auc(roc_nb), 2))),
    size = 4,
    color = "#48BFC4"  # R72 G191 U196
  ) +
  geom_text(
    aes(x = 0.8, y = 0.1, label = paste("AUC =", round(auc(roc_rf), 2))),
    size = 4,
    color = "#719DFF"  # R113 G157 U255
  ) +
  geom_text(
    aes(x = 0.8, y = 0.05, label = paste("AUC =", round(auc(roc_ensemble), 2))),
    size = 4,
    color = "#EC66E4"  # R236 G102 U228
  )

# 显示ROC曲线图
print(roc_plot)

## 混淆矩阵
cm1 <- confusionMatrix(as.factor(price_true), as.factor(price_pred_lr))
cm2 <- confusionMatrix(as.factor(price_true), as.factor(price_pred_svm))
cm3 <- confusionMatrix(as.factor(price_true), as.factor(price_pred_tree))
cm4 <- confusionMatrix(as.factor(price_true), as.factor(price_pred_nb))
cm5 <- confusionMatrix(as.factor(price_true), as.factor(price_pred_rf))
cm6 <- confusionMatrix(as.factor(price_true), as.factor(price_ensemble))

# 提取准确率
acc1 <- cm1$overall['Accuracy']
acc2 <- cm2$overall['Accuracy']
acc3 <- cm3$overall['Accuracy']
acc4 <- cm4$overall['Accuracy']
acc5 <- cm5$overall['Accuracy']
acc6 <- cm6$overall['Accuracy']

# 创建数据框
df <- data.frame(
  Method = c('logistic regression', 'support vector machine', 'decision tree','naive bayes','random forest','ensemble'),
  Accuracy = c(acc1, acc2, acc3, acc4, acc5, acc6)
)
df$Method <- factor(df$Method, levels = c('logistic regression', 'support vector machine', 'decision tree','naive bayes','random forest','ensemble'))

# 设cm_list是混淆矩阵列表
cm_list <- list(cm1, cm2, cm3, cm4, cm5, cm6)

# 初始化一个空的数据框来存储结果
df_f1 <- data.frame(Method = character(), F1Score = numeric())

# 对每个混淆矩阵计算F1-Score
for (i in 1:length(cm_list)) {
  cm <- cm_list[[i]]
  TP <- cm$table[2, 2]
  FP <- cm$table[2, 1]
  FN <- cm$table[1, 2]
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1Score <- 2 * (precision * recall) / (precision + recall)
  df_f1 <- bind_rows(df_f1, data.frame(Method = paste0("Method", i), F1Score = F1Score))
}

df_f1$Method <- c('logistic regression', 'support vector machine', 'decision tree','naive bayes','random forest','ensemble')

df_f1$Method <- factor(df$Method, levels = c('logistic regression', 'support vector machine', 'decision tree','naive bayes','random forest','ensemble'))

# Define the color palette
my_colors <- c("#7CAE00", "#00BFC4", "#F8766D", "#00A9FF", "#C77CFF", "#FF61CC")

# Plot for Accuracy
accuracy_plot <- ggplot(df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = 'identity', width = 0.5) +
  scale_fill_manual(values = my_colors) +
  coord_cartesian(ylim = c(0.4, 0.9)) +
  geom_text(aes(label = sprintf("%.2f", Accuracy)), vjust = -0.3, size = 3.5) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none",
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "#E5E5E5"),
    panel.grid.minor = element_blank()
  ) +
  labs(x = '', y = 'Accuracy', title = 'Accuracy of Different Methods')

# Plot for F1 Score without legend
f1_plot <- ggplot(df_f1, aes(x = Method, y = F1Score, fill = Method)) +
  geom_bar(stat = 'identity', width = 0.5) +
  scale_fill_manual(values = my_colors) +
  coord_cartesian(ylim = c(0.4, 0.9)) +
  geom_text(aes(label = sprintf("%.2f", F1Score)), vjust = -0.3, size = 3.5) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none",
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "#E5E5E5"),
    panel.grid.minor = element_blank()
  ) +
  labs(x = '', y = 'F1 Score', title = 'F1 Score of Different Methods')

# Create a dummy plot to extract the legend
legend_plot <- ggplot(df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = 'identity') +
  scale_fill_manual(values = my_colors, name = "Method") +
  guides(fill = guide_legend(override.aes = list(color = my_colors, fill = NA), nrow = 2)) +
  theme(legend.position = "bottom", legend.direction = "horizontal", legend.box = "vertical") +
  theme_minimal()

# Extract the legend
legend <- ggplotGrob(legend_plot)$grobs[[which(ggplotGrob(legend_plot)$layout$name == "guide-box")]]

# Arrange the plots and the legend
grid.arrange(
  arrangeGrob(accuracy_plot, f1_plot, ncol = 2, top = ""),
  legend,
  nrow = 2,
  heights = c(10, 1)  # Adjust the relative heights of the plots and legend
)

####################
## shiny交互界面
####################
price <- na.omit(as.numeric(data$price))

cleanliness <- data$Cleanliness.and.safety
service <- data$Services.and.conveniences
access <- data$Access
available <- data$Available.in.all.rooms
around <- data$Getting.around

cleanliness_text <- unlist(strsplit(cleanliness, "@@", fixed = TRUE))
service_text <- unlist(strsplit(service, "@@", fixed = TRUE))
access_text <- unlist(strsplit(access, "@@", fixed = TRUE))
available_text <- unlist(strsplit(available, "@@", fixed = TRUE))
around_text <- unlist(strsplit(around, "@@", fixed = TRUE))

# 去除包含NA的元素
cleanliness_text <- na.omit(cleanliness_text)
service_text <- na.omit(service_text)
access_text <- na.omit(access_text)
available_text <- na.omit(available_text)
around_text <- na.omit(around_text)

# 计算词频
wordfreq <- data.frame(table(c(cleanliness_text, service_text, access_text, available_text, around_text)))

# 绘制词云
wordfreq <- wordfreq[order(-wordfreq$Freq), ]

ui <- fluidPage(
  tags$head(
    tags$style(
      HTML("
        .title-center {
          text-align: center;
        }
        .text {
          font-size: 16px;
          font-family: 'Times New Roman';
          font-weight: normal;
        }
        .name {
          font-size: 16px;
          font-family: 'KaiTi';
          font-weight: normal;
           text-align: center; 
        }
        .shiny-wordcloud2-output {
          margin: 0 auto !important;
        }
        h2 {
        font-size: 20px !important;
      }
        
        
        }
      ")
    )
  ),
  div(class = "title-center", h2("Los Angeles Hotel Market Research and Pricing Guidance"),br()),
  
  div(class = "name"," By 姚圣泰 刘苏青 沈宇捷"),
  div(class = "text", br(),"This is a study on the Los Angeles hotel market. 
      You can start by exploring the price distribution across the entire market 
      in the first section, \"Price Distribution.\" Following that, 
      in the second section, the Wordcloud portion, 
      you can gather information about the facilities and 
      services offered by existing hotels. If you manage a hotel and 
      wish to set prices based on market conditions, you can input 
      information about your hotel's facilities, services, location, etc., 
      in the third section, \"Price Prediction.\" Different statistical methods 
      will be employed for price prediction. We hope you have a great experience!",br()),
  
  div(class = "title-center", h2("Part1--Price distribution"),br()),
  div(class = "text", "This is the price distribution of hotels in Los Angeles. 
      You can drag the slider to change the price interval. 
      Hovering over each bar will display the price range and the number of hotels 
      in that range. Additionally, you can zoom in on specific areas of the 
      chart by pressing and dragging the mouse over the desired region. 
      Double-clicking will reset the chart to its initial state.",br(),br()),
  
  sliderInput("intervalWidth", "Select Bin Width",
              min = 10, max = 150, step = 10,
              value = 10),
  
  # 绘制柱状图
  plotlyOutput("histogram"),
  
  div(class = "title-center", h2("Part2--Wordcloud"),br()),
  div(class = "text", "This is a word cloud regarding the cleanliness and safety, 
      services and conveniences, and access .etc provided by Los Angeles hotels. 
      You can hover your mouse over the corresponding words to see their respective 
      names and counts.",br()),
  
  wordcloud2Output("wordcloud_output"),
  
  
  div(class = "title-center", h2("Part3--Price Prediction"),br()),
  div(class = "text", "You can select the hotel's facilities, services, location information, etc. 
      in the selection box on the right side of the interface. 
      Only fill in the nearest distance in the location information below. 
      Choose the method on the left side of the interface, click the prediction button, 
      and you can see the predicted price.",br()),
  
  
  sidebarLayout(
    sidebarPanel(
      div(style = "margin-top: 160px; margin-bottom: 160px;",
          radioButtons("model_choice", "Select Model:",
                       choices = list("Linear Regression" = "linear",
                                      "Support Vector Regression" = "svm",
                                      "Random Forest Regression" = "rf")),
          actionButton("predict", "Predict"),
          hr(),
          h5("Predicted Price ($)", style = "font-size: 1.5em;"),
          wellPanel(textOutput("prediction"), style = "padding: 5px;")
      )
    ),
    mainPanel(
      tags$h3("Facilities"),
      fluidRow(
        column(4,
               selectInput("Language", 
                           "Language:",
                           choices = one_hot_encode(data_facilities$Languages.spoken)$unique_elements,
                           multiple = TRUE)),
        column(4,
               selectInput("ThingsToDo", 
                           "Things to Do:",
                           choices = one_hot_encode(data_facilities$Things.to.do..ways.to.relax)$unique_elements,
                           multiple = TRUE)),
        column(4,
               selectInput("CleanlinessSafety", 
                           "Cleanliness and Safety:",
                           choices = one_hot_encode(data_facilities$Cleanliness.and.safety)$unique_elements,
                           multiple = TRUE))
      ),
      fluidRow(
        column(6,
               selectInput("Dining", 
                           "Dining:",
                           choices = one_hot_encode(data_facilities$Dining..drinking..and.snacking)$unique_elements,
                           multiple = TRUE)),
        column(6,
               selectInput("ServicesConveniences", 
                           "Services and Conveniences:",
                           choices = one_hot_encode(data_facilities$Services.and.conveniences)$unique_elements,
                           multiple = TRUE))
      ),
      fluidRow(
        column(6,
               selectInput("Access", 
                           "Access:",
                           choices = one_hot_encode(data_facilities$Access)$unique_elements,
                           multiple = TRUE)),
        column(6,
               selectInput("AvailableItem", 
                           "Available Item:",
                           choices = one_hot_encode(data_facilities$Available.in.all.rooms)$unique_elements,
                           multiple = TRUE))
      ),
      tags$h3("Location"),
      fluidRow(
        column(6,
               numericInput("input1", "Airport (km)", value = 0)),
        column(6,
               numericInput("input2", "Station (km)", value = 0))
      ),
      fluidRow(
        column(6,
               numericInput("input3", "Hospital (km)", value = 0)),
        column(6,
               numericInput("input4", "Shopping (km)", value = 0))
      ),
      fluidRow(
        column(6,
               numericInput("input5", "Convenience Store (km)", value = 0)),
        column(6,
               numericInput("input6", "Cash Withdrawal (km)", value = 0))
      )
    )
  )
)

server <- function(input, output) {
  
  output$histogram <- renderPlotly({
    
    # 根据区间宽度创建 breaks
    breaks <- seq(0, max(price) + input$intervalWidth, by = input$intervalWidth)
    
    # 将价格分组到区间中
    price_intervals <- cut(price, breaks, include.lowest = TRUE)
    
    
    
    # 统计每个区间的频数
    freq_table <- table(price_intervals)
    
    # 将结果转换为数据框
    freq_df <- data.frame(Price_Range = names(freq_table), Frequency = as.numeric(freq_table),price_value=breaks[1:length(breaks)-1])
    
    # 将 Price_Range 列转为因子并保留原有顺序
    freq_df$Price_Range <- factor(freq_df$Price_Range, levels = levels(price_intervals))
    
    # 使用 plot_ly 绘制柱状图
    p <- plot_ly(data = freq_df, x = ~price_value, y = ~Frequency, type = "bar",
                 marker = list(color = "rgb(85,137,197)", line = list(color = "white", width = 0.5)),
                 hoverinfo = "text",
                 text = ~paste("Range:", Price_Range, "<br>Count:", Frequency))
    
    # 自定义布局
    p <- p %>% layout(title = "Price Distribution",
                      xaxis = list(title = "Price(USD)"),
                      yaxis = list(title = "Count"),
                      hoverlabel = list(bgcolor = "white", font = list(family = "Arial, sans-serif", size = 12)))
    
  })
  
  
  output$wordcloud_output <- renderWordcloud2({
    wordcloud2(wordfreq, size = 0.1, gridSize = 0, rotateRatio = 0, minRotation = -30, maxRotation = 30)
  })
  
  
  predictionEvent <- eventReactive(input$predict, {
    unique_elements <- c()
    for (item in facilities_vec) {
      unique_elements <- c(unique_elements, one_hot_encode(data_facilities[, item])$unique_elements)
    }
    selected_items <- c(
      input$Language, input$ThingsToDo, input$CleanlinessSafety,
      input$Dining, input$ServicesConveniences,
      input$Access, input$AvailableItem
    )
    
    n <- length(unique_elements)
    new_data_facilities <- rep(0, n)
    indices <- match(selected_items, unique_elements)
    new_data_facilities[indices] <- 1
    new_data_facilities <- t(new_data_facilities)%*%pca_result$rotation[, 1:200]
    
    new_data_locations <- rep(0, 8)
    new_data_locations[1] <- as.numeric(input$input1) * 1000
    new_data_locations[2] <- as.numeric(input$input2) * 1000
    new_data_locations[3] <- as.numeric(input$input3) * 1000
    new_data_locations[4] <- as.numeric(input$input4) * 1000
    new_data_locations[5] <- as.numeric(input$input5) * 1000
    new_data_locations[6] <- as.numeric(input$input6) * 1000
    new_data_locations[7] <- median(cleaned_data$V108)
    new_data_locations[8] <- median(cleaned_data$V109)
    
    new_data <- c(new_data_facilities, new_data_locations)
    new_data <- as.data.frame(t(new_data))
    colnames(new_data) <- colnames(cleaned_data[-1])
    
    if (input$model_choice == "linear") {
      pred <- round(predict(reg_lr, new_data), digits = 2)
    } else if (input$model_choice == "svm") {
      pred <- round(predict(reg_svr, new_data), digits = 2)
    } else if (input$model_choice == "rf") {
      pred <- round(predict(reg_rf, new_data), digits = 2)
    } else {
      pred <- NA
    }
    
    return(pred)
  })
  
  output$prediction <- renderText({
    pred <- predictionEvent()
  })
}

shinyApp(ui = ui, server = server)