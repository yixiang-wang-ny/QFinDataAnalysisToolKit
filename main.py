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

    pipe_runner = pipeline.QFinPipeLine()

    pipe_runner.add(
        pipeline.TestQFinPipe1().append(
            pipeline.TestQFinPipe2().append(
                [pipeline.TestQFinPipe3(), pipeline.TestQFinPipe4()]
            )
        )
    )

    pipe_runner.add(
        pipeline.TestQFinPipe5()
    )

    pipe_runner.train(session.data)



    return


if __name__ == '__main__':

    main()
