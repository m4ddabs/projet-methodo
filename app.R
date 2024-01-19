library(shiny)
library(tidyverse)
library(rjson)

model_list <- c("model_mlp_simple", 'model_mlp_3L', "model_rnn_simple", "model_cnn_simple", "model_mlp_4L", "model_rnn_gru_3L", "model_lstmbi_dense")
datasets <- read.table("datasets.txt")
datasets <- datasets$V1
dataset_info<-read_csv("infos_series_temporelles.csv")
dataset_info<-as.tbl(dataset_info)


# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Visualisation des des accuracy et loss des modèles"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
     selectInput("dataset",
                  "Dataset:",
                  datasets),
     
     selectInput("model",
                 "Model:",
                 model_list),
     
     radioButtons("rb", "Choose one:",
                  choiceNames = list(
                    "Accuracy", "Loss"
                  ),
                  choiceValues = list(
                    "acc", "loss"
                  )),
     
    ),
    
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot"),
      textOutput("description")
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$distPlot <- renderPlot({
    
    fp <- paste("resultats",input$dataset, paste0(input$model,".json"), sep="/")
    tb <- as_tibble(fromJSON(file = fp))
    
    
    if(input$rb == "acc"){
      ggplot(tb) + aes(x=1:tb$epochs[1]) +
        geom_line(aes(y=accuracy, color="accuracy") ,size=1) +
        geom_line(aes(y=val_accuracy, color ="val_accuracy"), size = 1) + 
        labs(title = "Accuracy du modèle",
             x = "Epochs",
             y = "Accuracy") +
        scale_color_manual(values = c("accuracy" = "blue", "val_accuracy" = "red"), labels = c("accuracy" = "training accuracy", "val_accuracy" = "validation accuracy")) +
        theme_minimal()
    }else{
      ggplot(tb) + aes(x=1:tb$epochs[1]) +
        geom_line(aes(y=loss, color="loss") ,size=1) +
        geom_line(aes(y=val_loss, color ="val_loss"), size = 1) + 
        labs(title = "Valeur de la fonction de la fonction de cout du modèle",
             x = "Epochs",
             y = "Loss") +
        scale_color_manual(values = c("loss" = "blue", "val_loss" = "red"), labels = c("loss" = "training loss", "val_loss" = "validation loss")) +
        coord_cartesian(ylim = c(min(tb$loss, tb$val_loss), tb$loss[1])) +
        theme_minimal()
    }
  })
  
  output$description <- renderText({
    fp2 <- paste("resultats",input$dataset, paste0(input$model,".json"), sep="/")
    tb2 <- as_tibble(fromJSON(file = fp2))

    train_size <- dataset_info |>
      filter(Data==input$dataset) |>
      select(Train_Size)
    paste(
      paste("Test Accuracy:", tb2$test_accuracy[1]),
      paste("Test Loss:", tb2$test_loss[1]),
      paste("Train Size:", train_size)
      ,sep =" ---------- ")

  })
}

# Run the application 
shinyApp(ui = ui, server = server)
