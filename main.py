"""
Arabic Sentiment ML Analyzer:
Develops and systematically evaluates multiple ML models for sentiment
classification in Arabic social media data.

METHODOLOGIES
1. Dataset is distributed amongst 4 sentiment classes
	    - POS, NEG, OBJ, NEUTRAL

2. Data preprocessing (clean/standardize data prior to feature extraction)
	a. text cleaning operations:
		-converting all chars to lowercase
		-remove HTML and URL tags
		-eliminate special chars and numerical tokens
	b. noise reduction:
		-correct common misspellings
		-expand contractions
	c. text normalization:
		-stemming or lemmatization
			-reduce words to their canonical form
		-removal of stop words irrelevant to semantic information
		-standardize text variations
			-dates
			-emojis

3. Feature Extraction:
	a. Automated Text Representations
		-TF-IDF: Counts how often words appear.
		 It gives more importance to unique, meaningful words
		 and less to common ones (like "the" or "and").

		-Word Embeddings (ex.,Word2Vec/FastText): Group words with similar meanings together.
		 words are related because they appear in similar contexts.

		-Contextual embeddings from Transformer Based Language Models (AI): The most advanced version.
		 These look at the whole sentence to understand the specific meaning of a word based on the words around it.

	b. Hand-Engineered (Human-Made) Features
	    Since Arabic is a complex language, manually tell the computer to look for specific clues to
	    capture linguistic characteristics specific to arabic

		-Dialect Indicators: Identifying if the tweet is in Modern Standard Arabic or a local dialect

		-Character Normalization: Cleaning up the text

		-Negation Handling: Marking words like "not" or "never," which completely change the meaning of a sentence.


	c. Extra Social Media "Clues"
	    These are the "bonus" details that help the AI understand the tone or intent of a tweet:
        -tweet length
        - word repetition patterns
        -punctuation usage
        -presence of emoticons or hashtags
		-Visuals: Looking at emojis/emoticons.
		-Structure: Checking the length of the tweet or if the user is repeating words for emphasis.
		-Metadata: Looking at hashtags and punctuation (like using lots of exclamation marks).

4. Model Training : training and evaluation of the following 3 classifiers:
	a. Decision trees (including random forests)
	b. Naive Bayes
	c. Neural Networks

5. Dataset splitting and Parameter Optimization:
	a. 60% of the dataset reserved for training
	b. 20% of the dataset reserved for validation
	c. 20% of the dataset reserved for testing
	d. parameter optimization for each classifier
	e. discuss challenges such as class imbalance and dataset sparsity
	f. propose methods such as resampling, weighting etc.)

6. Model Evaluation:
    a. accuracy
	b. precision
	c. recall
	d. F1-score
			"""

import pandas as pd
import matplotlib.pyplot as plt # Import pandas library for data manipulation

# Name of the input file consisting of the tweets in arabic
file_name = "Arabic.txt"

# Define the set of valid sentiment labels
labels = {"POS", "NEG", "OBJ", "NEUTRAL"}

# Create an empty list to store processed data
data = []

def main():
    data_analysis()


def convert_to_csv_file():
    # ensure that the input file exists and was found by the program
    try:
        # Open the text file in UTF-8 encoding
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip() # Remove leading/trailing spaces and newline characters
                if not line:
                    continue        # Skip empty lines

                # split the line from the right on the last tab character to separate text and label
                parts = line.rsplit("\t", 1)

                # Check if the split produced exactly two parts and the label is valid
                if len(parts) == 2 and parts[1] in labels:
                    text, label = parts         # Assign text and label
                    data.append([text, label])  # Add them to our data list
                else:
                    # skip malformed lines safely
                    print("Skipped:", line)

        # Convert the list of lists into a pandas DataFrame with columns "text" and "label"
        df = pd.DataFrame(data, columns=["text", "label"])

        # Save the structured DataFrame to a CSV file
        # utf-8-sig ensures Arabic characters are properly saved and can be opened in Excel
        df.to_csv(f"{file_name}.csv", index=False, encoding="utf-8-sig")

    # in case the input file doesn't exist or error arrises
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found. Please ensure it exists in the same directory as the Arabic Sentiment Analysis script.")


def data_analysis():
    convert_to_csv_file()
    df = pd.read_csv(f"{file_name}.csv")

    # Print the first 5 rows to check that everything looks correct
    print(df.head())
    print(df.shape)
    # Print the number of samples per sentiment label (class distribution)
    print(df["label"].value_counts())

    df["label"].value_counts().plot(kind="bar")
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
