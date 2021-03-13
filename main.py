from session import QFinDASession
import pipeline
import pandas as pd


def main():

    data_path = r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv'
    df = pd.read_csv(data_path)

    session = QFinDASession()
    session.add_data_from_data_frame(df, exclude_fields=['resp_1', 'resp_2', 'resp_3', 'resp_4', 'ts_id', 'weight'])
    session.data.set_time_series_id('date')
    session.data.set_target_columns('resp')

    pipe_line = pipeline.QFinPipeLine()

    pipe_line.add(
        pipeline.TestQFinPipe1(input_features=["feature_1", "feature_2", "feature_3", "feature_4"]).append(
            pipeline.TestQFinPipe2().append(
                [pipeline.TestQFinPipe3(input_features=["feature_1"]),
                 pipeline.TestQFinPipe4(input_features=["feature_2", "feature_3"])]
            )
        )
    )

    pipe_line.add(
        pipeline.TestQFinPipe5()
    )

    features = session.data.get_all_features()
    feature_out = pipe_line.train(features)

    return feature_out


if __name__ == '__main__':

    main()
