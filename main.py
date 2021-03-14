from session import QFinDASession
from pipeline import QFinPipeLine, PipeSelect
from transformation.missing_value_handler import FillWithMean
from transformation.scaler import MeanDeviationScaler
import pandas as pd


def main():

    data_path = r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv'
    df = pd.read_csv(data_path)

    session = QFinDASession()
    session.add_data_from_data_frame(df, exclude_fields=['resp_1', 'resp_2', 'resp_3', 'resp_4', 'ts_id', 'weight'])
    session.data.set_time_series_id('date')
    session.data.set_target_columns('resp')

    pipe_line = QFinPipeLine()

    features = session.data.get_all_features()
    factor_feature_names = ['feature_0']
    float_value_feature_names = ['feature_{}'.format(x) for x in range(1, len(features))]
    missing_value_pipe = FillWithMean(input_features=float_value_feature_names)
    scale_pipe_line = MeanDeviationScaler(input_features=float_value_feature_names)

    pipe_line.add([PipeSelect(input_features=factor_feature_names), missing_value_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names), scale_pipe_line])

    feature_out = pipe_line.train(features)

    return


if __name__ == '__main__':

    main()
