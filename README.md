# Aspect-Term-Polarity-Classification-in-Sentiment-Analysis

## Project Overview
This project implements an ABSA classifier for the DSBA Master's NLP Lecture 2024. In this project, I aim to develop a specialized Aspect-Based Sentiment Analysis (ABSA) classifier for the restaurant industry, designed to identify specific sentiments (positive, negative, or neutral) towards particular aspects within reviews, such as food quality or service. Unlike general sentiment analysis, ABSA provides a more detailed view by pinpointing the sentiment associated with specific targets within sentences, offering valuable insights into customer opinions. The classifier processes input data that includes the aspect category, target term, and full text, enabling the model to understand and analyze sentiment in context. The classifier uses DistilBERT, a lightweight version of BERT that retains much of BERT's performance but with fewer parameters and faster computation. DistilBERT is ideal for tasks requiring efficient and fast processing, the project addresses the challenges of context-aware sentiment classification while maintaining efficient computation and managing limited GPU memory, making it a practical tool for detailed opinion mining. 

## Authors
- Hanyun Hu

## Requirements
- Python 3.10.x
- PyTorch 2.1.x
- Transformers 4.34.1
- Tokenizers 0.14.1
- Datasets 2.14.5 
- Scikit-learn 1.2.1
- NumPy 1.26.0
- Pandas 2.1.1
- tqdm (for progress bars)

## Project Structure

- src/
    - classifier.py: ABSA Classifier implementation
    - tester.py: Script for testing the classifier

- data/
    - traindata.csv: Training dataset
    - devdata.csv: Development dataset 

## Classifier Implementation
### Model Architecture
- **Base Model:** DistilBERT (`distilbert-base-uncased`)
- **Task-specific layer:** Classification head for 3-way classification

### Data Handling
- **Custom ABSADataset class** for loading and preprocessing data
- **Input format:** The input consists of a sentence with an aspect term and its category formatted as: "Aspect: {aspect} Target: {target} Text: {sentence}". This format allows the model to focus on specific sentiment aspects of the text.
- **Tokenization:** The text is tokenized using DistilBERT's tokenizer, which splits the input into tokens and encodes them into numerical representations suitable for the model. It adds special tokens ([CLS] and [SEP]) and pads or truncates the sequence to a fixed maximum length (128 tokens in this case).
- **Label encoding:** negative (0), neutral (1), positive (2)

### Training Process
- **Optimizer:** AdamW with a learning rate of 2e-5
- **Early Stopping:** Prevents overfitting; monitors development set accuracy
- **Evaluation:** On the development set after each epoch

### Early Stopping
- Stops training if no improvement for 3 consecutive epochs
- Saves and loads the best model state

### Prediction
- Processes input data similarly to training
- Returns predicted sentiment labels for given instances

## Usage
1. Ensure all required libraries are installed.
2. Place the data files (`traindata.csv` and `devdata.csv`) in the `data/` directory.
3. Run the tester script:
   ```bash
   cd src
   python tester.py

## Results
The Aspect-Based Sentiment Analysis classifier achieved an accuracy range of 82.18% to 86.70% on the development set, demonstrating strong performance in predicting sentiment polarities for specific aspects in sentences. This accuracy was achieved using a fine-tuned DistilBERT model, which efficiently balanced performance. The early stopping mechanism typically led to convergence within 3-5 epochs, effectively preventing overfitting. 

## Conclusion
The ABSA classifier developed in this project shows good performance, achieving accuracy rates between 82.18% and 86.70% on the development set, demonstrating a successful balance between the complexity of aspect-based sentiment analysis and computational efficiency using the lightweight DistilBERT model. The use of early stopping enhances model generalization. While the model's results are promising, there is room for future enhancements, such as handling sentences with multiple aspects, exploring more advanced pre-trained models, and incorporating data augmentation techniques. Overall, the project not only meets its primary objectives but also establishes a solid foundation for advancing fine-grained sentiment analysis.
