from session import QFinDASession
import pandas as pd


def main():

    data_path = r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv'
    df = pd.read_csv(data_path)

    session = QFinDASession()
    session.add_data_from_data_frame(df)

    session.data_layer.get('weight')
    return


if __name__ == '__main__':

    main()
