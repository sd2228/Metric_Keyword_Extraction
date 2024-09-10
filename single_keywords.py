import pandas as pd
import regex as re
from keybert import KeyBERT
from tqdm import tqdm
import time
from settings import *


class KeywordExtractor:
    def __init__(self, pkl_file_path, output_folder):
        self.pkl_file_path = pkl_file_path
        self.output_folder = output_folder
        self.model = KeyBERT()

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def extract_doc_embeddings(self, df_month, pbar_year):
        keywords_list = []

        df_month['date'] = pd.to_datetime(df_month['date'], format='%Y%m%d')
        df_month = df_month.sort_values(by='date')

        for index, row in df_month.iterrows():
            text = row['text_sum']
            print(text)

            try:
                # Extract embeddings and keywords for a single post
                doc_embeddings, word_embeddings = self.model.extract_embeddings(
                    text, min_df=1, stop_words='english', keyphrase_ngram_range=(1, 3),
                )

                keywords = self.model.extract_keywords(
                    text, doc_embeddings=doc_embeddings, word_embeddings=word_embeddings,
                    top_n=10, min_df=1, stop_words='english', use_mmr=True, diversity=1,
                    keyphrase_ngram_range=(1, 3)
                )
                keywords_list.append(keywords)
            except ValueError as e:
                print(f"Error in extracting embeddings: {e}")
                keywords_list.append("no keywords found")

        pbar_year.update(1)

        df_month['keywords'] = keywords_list
        df_with_keywords = df_month[['id', 'date', 'keywords']].copy()

        file_name = f"{df_month['date'].dt.year.iloc[0]}_{df_month['date'].dt.month.iloc[0]}.csv"
        file_path = os.path.join(self.output_folder, file_name)
        df_with_keywords.to_csv(file_path, index=False)

        return  df_with_keywords, keywords, pbar_year


if __name__ == "__main__":
    pkl_file_path = DATA_DIR + "/UK_posts_clean_filtered.pkl"
    # Set paths
    input_dir = DATA_DIR
    output_folder = OUTPUT_DIR_SINGLE

    extractor = KeywordExtractor(pkl_file_path, output_folder)

    total_time_extract_keywords = 0
    total_time_extract_doc_embeddings = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            pkl_file_path = os.path.join(input_dir, filename)
            df = pd.read_pickle(pkl_file_path)
            df['date'] = pd.to_datetime(df['date'])
            available_years = df['date'].dt.year.unique()
            print("Available years in the DataFrame:", available_years)
            df_years = df[df['date'].dt.year.isin([2018, 2019, 2020, 2021, 2022])]
            for year in [2018, 2019, 2020, 2021, 2022]:
                with tqdm(total=12, desc=f'Year {year}') as pbar_year:
                    for month in range(1, 13):
                        df_month = df_years[(df_years['date'].dt.year == year) & (df_years['date'].dt.month == month)]
                        if df_month.empty:
                            pbar_year.update(1)
                            continue
                        # Limit processing to the first two rows per month for testing
                        # df_month = df_month.head(2)
                        try:
                            df_with_keywords, keywords, pbar_year = extractor.extract_doc_embeddings(df_month, pbar_year)
                        except ValueError as e:
                            print(f"Error in processing {year}-{month}: {e}")
                            pbar_year.update(1)
                            continue


    # Optionally, you can measure and print total times here if needed
    print(f"Total time for extracting keywords: {total_time_extract_keywords} seconds")
    print(f"Total time for extracting document embeddings: {total_time_extract_doc_embeddings} seconds")