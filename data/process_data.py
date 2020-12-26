import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the datasets from the given file paths

    Parameters
    ----------
    messages_filepath, categories_filepath : str
        The path of the CSV files

    Returns
    -------
    df
        The dataframe that is made by merging the 2 csv files
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
    """Cleand the dataframe returned by load_data function,
    by removing duplicates and converting the categories into columns

    Parameters
    ----------
    df : DataFrame
        The dataframe returned by load_data function,

    Returns
    -------
    df
        The cleaned dataframe
    """

    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop(['categories'],axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    df = df.replace(2, 0)
    return df


def save_data(df, database_filename):
    """saves the dataframe into a database

    Parameters
    ----------
    df : DataFrame
        The dataframe returned by clean_data function
    database_filename : The file path where the database is to be created

    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    """
    The main function sequentiall calls the above functions to create a cleaned
    database of the povided messages and categories csv files
    """

    if len(sys.argv) == 4:
        #Reading in the filepaths of various files
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
