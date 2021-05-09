// ===========================
// BlazorSpread.net
// ===========================
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;

namespace BlazorServerML.ML
{
    public class Trainer
    {
        public static readonly string
            TRAIN_FILE = "housing.csv",
            TRAIN_PATH = Startup.PATH + "/Data/" + TRAIN_FILE,
            MODEL_FILE = "TrainedModel.zip",
            MODEL_PATH = Startup.PATH + "/Data/" + MODEL_FILE;

        readonly MLContext _ml = new(seed: 1);

        #region Prompt
        public delegate Task PromptHandler(string message);
        public PromptHandler Prompt;
        #endregion

        // quality control
        double r2Average;
        const double ACCEPTED_ACCURACY = 0.7;

        public async Task CreateModel(string trainFile = null)
        {
            if (trainFile == null) {
                trainFile = TRAIN_PATH;
            }
            await Echo($"Processing file: {Path.GetFileName(trainFile)} | " +
                       $"{new FileInfo(trainFile).Length:#,###,###} Bytes");

            await Echo ("Loading data...");
            var trainingDataView = _ml.Data.LoadFromTextFile<HousingData>(
                path: trainFile,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            await Echo ("Building pipeline...");
            var trainingPipeline = BuildTrainingPipeline();

            await Echo("Training model...");
            var model = trainingPipeline.Fit(trainingDataView); 

            await Echo ("Evaluating model...");
            await Evaluate(trainingDataView, trainingPipeline);

            await Echo ("Conclusion");
            if (r2Average < ACCEPTED_ACCURACY) {
                await Echo ($"The trained model has low accuracy, less than {ACCEPTED_ACCURACY}, and will not be published.");
            }
            else {
                await Echo($"\nThe trained model has acceptable accuracy and will be published.");
                await Echo ("Saving the model...");
                _ml.Model.Save(model, trainingDataView.Schema, MODEL_PATH);
                await Echo ($"Model file was publishd as {MODEL_FILE} | {new FileInfo(MODEL_PATH).Length:#,###,###} Bytes");
            }
            await Echo ("End of process");
        }

        IEstimator<ITransformer> BuildTrainingPipeline()
        {
            var feactures = new string[] {
                "OceanProximity",
                "Longitude",
                "Latitude",
                "HousingMedianAge",
                "TotalRooms",
                "TotalBedrooms",
                "Population",
                "Households",
                "MedianIncome"
            };
            var inputOutputColumns = new InputOutputColumnPair("OceanProximity", "OceanProximity");

            // data process configuration with pipeline data transformations 
            var dataProcessPipeline = _ml
                .Transforms.Categorical.OneHotEncoding(new[] { inputOutputColumns })
                .Append(_ml.Transforms.Concatenate("Features", feactures));

            // set the training algorithm 
            var trainer = _ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() {
                NumberOfIterations = 200,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            });

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        async Task Evaluate(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            await Echo("\nCross-validating to get model's accuracy metrics");
            var crossValidationResults = _ml.Regression.CrossValidate(
                trainingDataView,
                trainingPipeline,
                numberOfFolds: 5,
                labelColumnName: "Label");
            var l1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var l2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var rms = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var r2 = crossValidationResults.Select(r => r.Metrics.RSquared);
            // QC
            r2Average = r2.Average();
            // report 
            await Echo($"Metrics for Regression model");
            await Echo($"Mean Absolute Error:     {l1.Average():0.###}");
            await Echo($"Mean Squared Error:      {l2.Average():0.###}");
            await Echo($"Root Mean Squared Error: {rms.Average():0.###}");
            await Echo($"Average Loss Function:   {lossFunction.Average():0.###}");
            await Echo($"Average R-squared:       {r2.Average():0.###}\n");
        }

        async Task Echo(string message)
        {
            Prompt?.Invoke(message + "\n");
            // asynchronous for the reactivity of the interface that receives the event
            await Task.Delay(100);
        }
    }
}
