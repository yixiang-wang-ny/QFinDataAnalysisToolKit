from session import QFinDASession
from pipeline import QFinPipeLine, PipeSelect
from transformation.missing_value_handler import FillWithMean
from transformation.scaler import MeanDeviationScaler
from transformation.singular_value_decomposer import PCA
from model.generative_additive_model import GAM
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
    factor_feature_names = [x.name for x in features if x.is_factor()]
    float_value_feature_names = [x.name for x in features if not x.is_factor()]

    missing_value_pipe = FillWithMean(input_features=float_value_feature_names)
    scale_pipe = MeanDeviationScaler(input_features=float_value_feature_names)
    pca_pipe = [PCA(input_features=['feature_{}'.format(x) for x in range(s, e+1)]) for s, e in FEATURE_SPEC_SET]

    pipe_line.add([PipeSelect(input_features=factor_feature_names), missing_value_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names), scale_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names)]+pca_pipe)

    session.set_feature_transformer(pipe_line)
    session.run_feature_transformer()

    rolling_window_generator = session.data.get_rolling_window_generator(train_window_size=20, test_window_size=5,
                                                                         step=20)

    for data in rolling_window_generator:

        gam = GAM()
        gam.train(data.train_in, data.train_out)
        gam.predict(data.test_in)

    return


if __name__ == '__main__':

    main()
