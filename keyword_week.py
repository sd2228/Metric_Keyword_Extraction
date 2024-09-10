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

    def extract_doc_embeddings(self, df_month):
        # Group posts by week
        df_month['date'] = pd.to_datetime(df_month['date'], format='%Y%m%d')
        df_month = df_month.sort_values(by='date')
        df_month['week'] = pd.to_datetime(df_month['date'], format='%Y%m%d').dt.strftime('%W')
        print(df_month['week'].unique())

        df_weekly = df_month.groupby('week')

        weekly_embeddings = {}

        for week, df_week in tqdm(df_weekly, desc='Computing doc embeddings per week'):
            week_text = ' '.join(df_week['text_sum'].tolist())
            doc_embeddings, word_embeddings = self.model.extract_embeddings(
                [week_text], min_df=1, stop_words='english', keyphrase_ngram_range=(1, 3),
            )

            weekly_embeddings[week] = doc_embeddings
        return weekly_embeddings, df_month

    # def extract_word_embeddings(self, df_month):
    #     word_embeddings_list = []
    #
    #     for post in tqdm(df_month['text_sum'], desc='Computing word embeddings'):
    #         doc_embeddings, word_embeddings = self.model.extract_embeddings([post], min_df=1, stop_words='english',
    #                                                                         keyphrase_ngram_range=(1, 3),
    #                                                                         )
    #         word_embeddings_str = [','.join(map(str, emb)) for emb in word_embeddings]
    #         word_embeddings_list.append(word_embeddings_str)
    #
    #     df_month['word_embeddings'] = word_embeddings_list
    #     return word_embeddings_list, df_month

    def extract_keywords(self, weekly_embeddings, df_month, year, month):
        keywords_list = []

        for index, row in tqdm(df_month.iterrows(), total=len(df_month), desc='Computing keywords'):
            post = row['text_sum']
            week = row['week']
            # word_embeddings = row['word_embeddings']
            # word_embeddings_array = np.array(word_embeddings)
            if week in weekly_embeddings:
                doc_embedding_week = weekly_embeddings[week]
                # print(doc_embedding_week)
                try:
                    keywords = self.model.extract_keywords([post], doc_embeddings=doc_embedding_week,
                                                           top_n=10, min_df=1, stop_words='english', use_mmr=True,
                                                           diversity=1,
                                                           keyphrase_ngram_range=(1, 3)
                                                           )
                    keywords_list.append(keywords)
                except ValueError as e:
                    print(f"Error in extracting keywords: {e}")
                    keywords_list.append("no keywords found")

                    continue

            else:
                print(f"No embeddings found for week {week}")
                keywords_list.append("no keywords found")

        df_month['keywords'] = keywords_list
        df_with_keywords = df_month[['id', 'date', 'keywords']].copy()
        print(df_with_keywords.head())

        #save it as csv
        file_name = f"{month}_{year}.csv"
        file_path = os.path.join(output_folder, file_name)
        df_with_keywords.to_csv(file_path, index=False)

        # salva doc_embeddings AAAA_MM_PRIMOGIORNO_ULTIMO_GIORNO 2022_01_18-2022_01_24 (lunedi-domenica) - CSV
        # ogni settimana csv con nome AAAA_MM_PRIMOGIORNO_ULTIMO_GIORNO e doc embedding del singolo post
        # ID DATE EMBEDDINGS del singolo post

        return keywords, df_month, df_with_keywords


if __name__ == "__main__":
    pkl_file_path = DATA_DIR + "/UK_posts_clean_filtered.pkl"
    # Set paths
    input_dir = DATA_DIR
    output_folder = OUTPUT_DIR_WEEK

    extractor = KeywordExtractor(pkl_file_path, output_folder)

    # test for one month
    # year = 2022
    # month = 5
    total_time_extract_keywords = 0
    total_time_extract_doc_embeddings = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            pkl_file_path = os.path.join(input_dir, filename)
            df = pd.read_pickle(pkl_file_path)
            # df = df.dropna()
            df['date'] = pd.to_datetime(df['date'])
            available_years = df['date'].dt.year.unique()
            print("Available years in the DataFrame:", available_years)
            df_years = df[df['date'].dt.year.isin([2018, 2019, 2020, 2021, 2022])]
            for year in [2018,2019,2020,2021,2022]: #2019, 2020, 2021, 2022
                for month in range(1, 13): #13
                    df_month = df_years[(df_years['date'].dt.year == year) & (df_years['date'].dt.month == month)]
                    # df_sample = df_month.sample(n=2000, random_state=42)
                    # print(df_sample.date.unique())

                    # Measure time for extract_doc_embeddings
                    start_time = time.time()
                    try:
                        weekly_embeddings_dictionary, df_month_doc = extractor.extract_doc_embeddings(df_month)
                    except ValueError as e:
                        print(f"Error in extracting document embeddings: {e}")
                        continue
                    end_time = time.time()
                    total_time_extract_doc_embeddings += end_time - start_time
                    print(total_time_extract_doc_embeddings)

                    # Measure time for extract_keywords
                    start_time = time.time()
                    try:
                        keywords, df_month, df_with_keywords = extractor.extract_keywords(weekly_embeddings_dictionary,
                                                                                          df_month_doc, year, month)
                    except ValueError as e:
                        print(f"Error in extracting keywords: {e}")
                        continue
                    end_time = time.time()
                    total_time_extract_keywords += end_time - start_time
                    print(total_time_extract_keywords)


            # weekly_embeddings_dictionary, df_sample_doc = extractor.extract_doc_embeddings(df_sample)
            # keywords, df_month, df_with_keywords = extractor.extract_keywords(weekly_embeddings_dictionary, df_sample_doc,year, month)

    # LAUNCH ON ALL
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".pkl"):
    #         pkl_file_path = os.path.join(input_dir, filename)
    #         df = pd.read_pickle(pkl_file_path)
    #         df['date'] = pd.to_datetime(df['date'])
    #         available_years = df['date'].dt.year.unique()
    #         print("Available years in the DataFrame:", available_years)
    #         df_years = df[df['date'].dt.year.isin([2018, 2019, 2020, 2021, 2022])]
    #         for year in [2018, 2019, 2020, 2021, 2022]:
    #             for month in range(1, 13):
    #                 df_month = df_years[(df_years['date'].dt.year == year) & (df_years['date'].dt.month == month)]
    #                 df_sample = df_month.sample(n=10, random_state=42)
    #                 print(df_sample.date.unique())
    #                 weekly_embeddings = extractor.extract_doc_embeddings(df_sample)