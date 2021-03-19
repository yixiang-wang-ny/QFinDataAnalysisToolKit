from session import QFinDASession
from pipeline import QFinPipeLine, PipeSelect
from transformation.missing_value_handler import FillWithMean
from transformation.scaler import MeanDeviationScaler
from transformation.singular_value_decomposer import PCA
from model.generative_additive_model import GAM
from model.criteria import DirectionalAccuracy
from model.ensemble import DirectionalVotes


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

    session = QFinDASession()
    session.add_data_from_csv(
        r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv',
        exclude_fields=['resp_1', 'resp_2', 'resp_3', 'resp_4', 'ts_id', 'weight'],
        factor_fields=('feature_0', )
    )
    session.data.set_time_series_id('date')
    session.data.set_target_fields('resp')

    features = session.data.get_all_features()
    factor_feature_names = [x.name for x in features if x.is_factor()]
    float_value_feature_names = [x.name for x in features if not x.is_factor()]

    # set up pipe line
    pipe_line = QFinPipeLine()

    missing_value_pipe = FillWithMean(input_features=float_value_feature_names)
    scale_pipe = MeanDeviationScaler(input_features=float_value_feature_names)
    pca_pipe = [PCA(input_features=['feature_{}'.format(x) for x in range(s, e+1)]) for s, e in FEATURE_SPEC_SET]

    pipe_line.add([PipeSelect(input_features=factor_feature_names), missing_value_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names), scale_pipe])
    pipe_line.add([PipeSelect(input_features=factor_feature_names)]+pca_pipe)

    session.set_feature_transformer(pipe_line)

    # get data generator
    rolling_window_generator = session.data.get_rolling_window_generator(
        train_window_size=20, test_window_size=5, step=100
    )
    session.set_data_validation_generator(rolling_window_generator)

    # add models and search
    session.add_model_config(GAM, lam=25000)
    session.add_model_performance_measure(DirectionalAccuracy())
    session.search_models()

    summary = session.get_trained_model_summary()
    print(summary)

    rolling_window_generator2 = session.data.get_rolling_window_generator(
        train_window_size=20, test_window_size=5, step=100
    )

    wrapped_gam = DirectionalVotes.wrap(GAM, lam=25000)
    wrapped_gam.train()

    for data in rolling_window_generator2:
        wrapped_gam.train(data.train_in, data.train_out)

    return


if __name__ == '__main__':

    main()
