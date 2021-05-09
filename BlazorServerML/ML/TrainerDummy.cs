﻿using System.IO;
using System.Threading.Tasks;

namespace BlazorServerML.ML
{
    public class TrainerDummy
    {
        public static readonly string
            TRAIN_FILE = "housing.csv",
            TRAIN_PATH = Startup.PATH + "/Data/" + TRAIN_FILE,
            MODEL_FILE = "TrainedModel.zip",
            MODEL_PATH = Startup.PATH + "/Data/" + MODEL_FILE;

        #region Prompt
        public delegate Task PromptHandler(string message);
        public PromptHandler Prompt;
       
        #endregion

        public async Task CreateModel(string trainFile = null)
        {
            if (trainFile == null) {
                trainFile = TRAIN_PATH;
            }

            await Echo($"Processing file: {Path.GetFileName(trainFile)} | {new FileInfo(trainFile).Length:#,###,###} Bytes");

            await Echo("Loading data...");

            await Echo("Building pipeline...");
            // 2. Build training pipeline

            await Echo("Training model...");
            // 3. Train Model

            await Echo("Evaluating model...");
            // 4. Evaluate quality of Model

            await Echo("Conclusion");
            // 5. Conclution

            await Echo("End of process");
        }

        async Task Echo(string message)
        {
            Prompt?.Invoke(message + "\n");

            await Task.Delay(1000);
        }
    }
}
