library(shiny)
library(tidyverse)

model_list <- c("model_mlp_simple", 'model_mlp_3L', "model_rnn_simple", "model_cnn_simple", "model_mlp_4L", "model_rnn_gru_3L", "model_lstmbi_dense")
datasets <- read.table("datasets.txt")
datasets <- datasets$V1

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Old Faithful Geyser Data"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
     selectInput("dataset",
                  "Dataset:",
                  datasets),
     
     selectInput("model",
                 "Model:",
                 model_list),
     
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$distPlot <- renderPlot({
    
    fp <- paste("resultats",input$dataset, paste0(input$model,".json"), sep="/")
    tb <- as_tibble(fromJSON(file = fp))
    
    
    ggplot(tb) + aes(x=1:tb$epochs[1]) +
      geom_line(aes(y=accuracy, color="accuracy") ,size=1) +
      geom_line(aes(y=val_accuracy, color ="val_accuracy"), size = 1) + 
      labs(title = "Line Plot of Two Columns",
           x = "Epochs",
           y = "Accuracy") +
      scale_color_manual(values = c("accuracy" = "blue", "val_accuracy" = "red"), labels = c("accuracy" = "training accuracy", "val_accuracy" = "validation accuracy")) +
      theme_minimal()
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
