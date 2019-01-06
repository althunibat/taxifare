using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Transforms.Normalizers;
using StackExchange.Redis;

namespace TaxiFare.Trainer
{
    internal class Program
    {
        private static ConnectionMultiplexer Redis;

        private static async Task Main(string[] args)
        {
            if (!await IsValid(args)) return;
            var trainingDataFile = args[0];
            var testDataFile = args[1];
            Redis = ConnectionMultiplexer.Connect(args[2]);
            //Create ML Context with seed for repeteable/deterministic results
            var mlContext = new MLContext(0);

            // STEP 1: Common data loading configuration
            var (testDataView, trainingDataView) = PerformStep1(mlContext, trainingDataFile, testDataFile);

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = PerformStep2(mlContext);

            // STEP 3: Set the training algorithm, then create and config the modelBuilder - Selected Trainer (SDCA Regression algorithm)                            
            var trainingPipeline = PerformStep3(mlContext, dataProcessPipeline);

            // STEP 4: Train the model fitting to the DataSet
            //The pipeline is trained on the data set that has been loaded and transformed.
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // STEP 5: Evaluate the model and show accuracy stats
            await PerformStep5(trainedModel, testDataView, mlContext);

            // STEP 6: Save/persist the trained model to a .ZIP file
            PerformStep6(trainedModel, mlContext);

            await Console.Out.WriteLineAsync("done!");
            Console.ReadLine();
        }

        private static void PerformStep6(TransformerChain<RegressionPredictionTransformer<LinearRegressionPredictor>> trainedModel, IHostEnvironment mlContext)
        {
            byte[] bytes;
            using (var stream = new MemoryStream())
            {
                trainedModel.SaveTo(mlContext, stream);
                bytes = stream.ToArray();
            }

            Redis.GetDatabase(1).StringSet("tm", bytes);
        }

        private static async Task PerformStep5(TransformerChain<RegressionPredictionTransformer<LinearRegressionPredictor>> trainedModel, IDataView testDataView, MLContext mlContext)
        {
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            await PrintRegressionMetrics("", metrics);
        }

        private static EstimatorChain<RegressionPredictionTransformer<LinearRegressionPredictor>> PerformStep3(MLContext mlContext, EstimatorChain<ITransformer> dataProcessPipeline)
        {
            var trainer =
                mlContext.Regression.Trainers.StochasticDualCoordinateAscent("Label", "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }

        private static EstimatorChain<ITransformer> PerformStep2(MLContext mlContext)
        {
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId", "VendorIdEncoded"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode", "RateCodeEncoded"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType", "PaymentTypeEncoded"))
                .Append(mlContext.Transforms.Normalize("PassengerCount",
                    mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize("TripTime",
                    mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize("TripDistance",
                    mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded",
                    "PassengerCount", "TripTime", "TripDistance"));
            return dataProcessPipeline;
        }

        private static (IDataView testDataView, IDataView trainingDataView) PerformStep1(MLContext mlContext,
            string trainingDataFile, string testDataFile)
        {
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6)
                }
            });
            var baseTrainingDataView = textLoader.Read(trainingDataFile);
            var testDataView = textLoader.Read(testDataFile);


            //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data
            var trainingDataView =
                mlContext.Data.FilterByColumn(baseTrainingDataView, "FareAmount", 1, 150);
            return (testDataView, trainingDataView);
        }

        private static async Task<bool> IsValid(IReadOnlyList<string> args)
        {
            // args of zero is the training data path.
            if (args == null || args.Count < 3)
            {
                await Console.Out.WriteLineAsync("please pass training & test data files (csv) paths");
                return false;
            }
            if (!File.Exists(args[0]))
            {
                await Console.Out.WriteLineAsync($"training data file (csv) path is invalid: {args[0]}");
                return false;
            }
            if (!File.Exists(args[1]))
            {
                await Console.Out.WriteLineAsync($"test data file (csv) path is invalid: {args[1]}");
                return false;
            }
            return true;
        }

        public static async Task PrintRegressionMetrics(string name, RegressionEvaluator.Result metrics)
        {
            await Console.Out.WriteLineAsync($"*************************************************");
            await Console.Out.WriteLineAsync($"*       Metrics for {name} regression model      ");
            await Console.Out.WriteLineAsync($"*------------------------------------------------");
            await Console.Out.WriteLineAsync($"*       LossFn:        {metrics.LossFn:0.##}");
            await Console.Out.WriteLineAsync($"*       R2 Score:      {metrics.RSquared:0.##}");
            await Console.Out.WriteLineAsync($"*       Absolute loss: {metrics.L1:#.##}");
            await Console.Out.WriteLineAsync($"*       Squared loss:  {metrics.L2:#.##}");
            await Console.Out.WriteLineAsync($"*       RMS loss:      {metrics.Rms:#.##}");
            await Console.Out.WriteLineAsync($"*************************************************");
        }
    }
}
