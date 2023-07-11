#
# # Integration test for the model
# import os
#
# from click.testing import CliRunner
#
# from src.main import predict, train
#
#
# def test_train():
#     path = "models/test_model.bin"
#     runner = CliRunner()
#     result = runner.invoke(
#         train, ["--data", "data/imdb_small.csv", "--path", f"{path}"]
#     )
#     assert result.exit_code == 0
#     assert os.path.exists("models/test_model.bin")
#     assert os.path.getsize("models/test_model.bin") > 0
#     assert (
#         result.output == f"Trained the model successfully\nSaved the model to {path}\n"
#     )
#     os.remove("models/test_model.bin")
#
#
# def test_predict():
#     runner = CliRunner()
#     result = runner.invoke(predict, ["--review", "This movie was great!"])
#     assert result.exit_code == 0
#     assert (
#         result.output
#         == "The prediction is was successful\nSentiment of the review: 1\n"
#     )
#     assert os.path.exists("data/predictions.csv")
