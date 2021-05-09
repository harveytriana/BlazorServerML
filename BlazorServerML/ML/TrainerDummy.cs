using System.IO;

namespace BlazorServerML.ML
{
    public class TrainerDummy
    {
        public static readonly string
            TRAIN_FILE = "housing.csv",
            TRAIN_PATH = Startup.PATH + "/Data/" + TRAIN_FILE;

        #region Prompt
        public delegate void PromptHandler(string message);
        public PromptHandler Prompt;
        void Echo(string message) => Prompt?.Invoke(message + "\n");
        #endregion

        public void CreateModel(string trainFile = null)
        {
            if (trainFile == null) {
                trainFile = TRAIN_PATH;
            }

            Echo($"Processing file: {Path.GetFileName(trainFile)} | {new FileInfo(trainFile).Length:#,###,###} Bytes");

            Echo("Loading data...");

            Echo("Building pipeline...");
            // 2. Build training pipeline

            Echo("Training model...");
            // 3. Train Model

            Echo("Evaluating model...");
            // 4. Evaluate quality of Model

            Echo("Conclusion");
            // 5. Conclution

            Echo("End of process");
        }

    }
}
