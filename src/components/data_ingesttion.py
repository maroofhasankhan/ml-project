# Import required libraries
import os  # For operating system operations like file paths
import sys  # For system-specific parameters and functions
from src.logger import logging  # Import custom logging functionality
from src.exception import CustomException  # Import custom exception handling
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass  # For creating data classes


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths"""
    # Define paths for storing train, test and raw data files in artifacts directory
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv') 
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """Class for handling all data ingestion operations"""
    def __init__(self):
        # Initialize with configuration settings
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to perform data ingestion process:
        1. Read the data
        2. Save raw data
        3. Split into train and test sets
        4. Save train and test sets
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split data into training and test sets (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return paths of generated train and test datasets
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If any error occurs, raise custom exception
            raise CustomException(e, sys)


