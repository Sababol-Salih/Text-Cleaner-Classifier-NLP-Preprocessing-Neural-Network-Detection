Feedforward AI – Text Cleaner & Sentiment Classifier  
----------------------------------------------------

📌 Project Description:
This is a prototype sentiment classification system built for FeedForward AI. It reads customer feedback, cleans and processes the text using NLP techniques, vectorizes it using Bag of Words, and then classifies it using a simple feedforward neural network implemented in TensorFlow/Keras.

✅ Key Features:
- Cleans and normalizes raw customer feedback
- Converts text into Bag of Words vectors
- Trains a feedforward neural network to detect sentiment (positive or negative)
- Allows live predictions from user input via console

----------------------------------------------------
🧹 Text Preprocessing Steps:
The following steps are applied to clean each feedback message:
Convert to lowercase
2. Remove punctuation and extra whitespaces
3. Tokenize text into words
4. Remove stopwords using NLTK
5. Lemmatize each word (convert to base form)

Example:
Raw: “This product is absolutely amazing!!!”
Cleaned: “product absolutely amazing”

----------------------------------------------------
🧠 Bag of Words Construction:
- We use Count Vectorizer from scikit-learn to create a vocabulary from the cleaned feedback text.
- Each cleaned message is converted into a sparse vector based on word frequency.
- These vectors are used as input features for the neural network.

----------------------------------------------------
🔧 Neural Network Design:
- Input Layer: Number of input features = vocabulary size
- Hidden Layer: Dense(16), activation='relu'
- Output Layer: Dense(1), activation='sigmoid'

Training Parameters:
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs : 20
- Batch Size :4

----------------------------------------------------
📊 Model Evaluation:
- The dataset is split into training and testing sets.
- Model is evaluated using accuracy on the test set.
- Accuracy score is printed after training.

----------------------------------------------------
📁 Files Included:
- text_cleaner.py – Text cleaning functions using NLTK
- training.py – Main script to train the sentiment classifier
- feedback.csv – Sample labeled feedback (1 = Positive, 0 = Negative)
- predict_loop.py – Optional console input script to test live predictions
- README.txt – This file

----------------------------------------------------
▶️ Demo Instructions (Video):
In the short demo video, you should show:
Running training.py to load, clean, vectorize, and train the model
2. The model’s accuracy score after training
3. Running predict_loop.py to enter new feedback manually and see the sentiment result

----------------------------------------------------
👨‍💻 Developed by: sababol salih
For FeedForward AI – NLP Classification Project
