from session import QFinDASession
from pipeline import QFinPipeLine, PipeSelect
from transformation.missing_value_handler import FillWithMean
from transformation.scaler import MeanDeviationScaler
from transformation.singular_value_decomposer import PCA
import pandas as pd


FEATURE_SPEC_SET = (
    (1, 2),
    (3, 6),
    (7, 8),
    (9, 16),
    (17, 40),
    (41, 53),
    (54, 54),
    (55, 59),
    (60, 68),
    (69, 71),
    (72, 119),
    (120, 129)
)


def main():

    data_path = r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv'
    df = pd.read_csv(data_path)

    session = QFinDASession()
    session.add_data_from_data_frame(df, exclude_fields=['resp_1', 'resp_2', 'resp_3', 'resp_4', 'ts_id', 'weight'],
                                     factor_fields=('feature_0', ))
    session.data.set_time_series_id('date')
    session.data.set_target_fields('resp')

    pipe_line = QFinPipeLine()

    features = session.data.get_all_features()
    factor_feature_names = ['feature_0']
    float_value_feature_names = ['feature_{}'.format(x) for x in range(1, len(features))]

    missing_value_pipe = FillWithMean(input_features=float_value_feature_names)
    scale_pipe = MeanDeviationScaler(input_features=float_value_feature_names)
    pca_pipe = [PCA(input_features=['feature_{}'.format(x) for x in range(s, e+1)]) for s, e in FEATURE_SPEC_SET]

    pipe_line.add([PipeSelect(input_features=factor_feature_names), missing_value_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names), scale_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names)]+pca_pipe)

    feature_out = pipe_line.train(features)

    rolling_window_generator = session.data.get_rolling_window_generator()

    for data in rolling_window_generator:
        print("get some data")

    return


if __name__ == '__main__':

    main()
