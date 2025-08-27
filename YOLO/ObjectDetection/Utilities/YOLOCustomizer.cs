using System.IO;
#if UNITY_6000_2_OR_NEWER
using Unity.InferenceEngine;
#else
using Unity.Sentis;
#endif
using UnityEngine;

namespace YOLOTools.YOLO.ObjectDetection.Utilities
{
    public class YOLOCustomizer
    {
        private static bool _addNMS = false;

        private static float _iouThreshold = 0.5f;
        private static float _scoreThreshold = 0.5f;

        public static bool CustomizeModel(ModelAsset modelAsset, YOLOCustomizationParameters parameters, out YOLOModel yoloModel)
        {
            try
            {
                var model = ModelLoader.Load(modelAsset);

                _iouThreshold = parameters.IoUThreshold;
                _scoreThreshold = parameters.ScoreThreshold;
                _addNMS = parameters.AddNMS;

                if (parameters.AddClassificationHead) AddClassificationHead(ref model);

                if (parameters.QuantizeModel) ModelQuantizer.QuantizeWeights(parameters.QuantizationType, ref model);

                var yoloModelDir = Directory.CreateDirectory(Path.Join(Application.persistentDataPath, "YOLOModels"));
                var customDir = Directory.CreateDirectory(Path.Join(yoloModelDir.FullName, "Custom"));

                var newFilePath = Path.Join(customDir.FullName, $"{modelAsset.name}.sentis");

                ModelWriter.Save(newFilePath, model);

                yoloModel = new YOLOModel(ModelLoader.Load(newFilePath));
            }
            catch
            {
                Debug.Log("YOLOModelCustomization failed");
                yoloModel = null;
                return false;
            }


            return true;
        }

        private static void AddClassificationHead(ref Model model)
        {
            var graph = new FunctionalGraph();

            var inputs = graph.AddInputs(model);
            var output = Functional.Forward(model, inputs)[0];

            /*
            Output Format

            1x84xN tensor

            Each of the N columns are a prediction. The first 0..4 rows of each column are the position and scale. The last 4..84 rows are the confidences of each of the 80 classes.

            */


            var slicedClasses = output[.., 4..84, ..]; // (1,80,N)

            var argMaxClasses = Functional.ArgMax(slicedClasses, 1); // (1,N)
            var confidences = Functional.ReduceMax(slicedClasses, 1); // (1,N)
            var slicedPositions = output[.., 0..4, ..]; // (1,4,N)

            FunctionalTensor coords;
            FunctionalTensor classIds;
            FunctionalTensor scores;

            if (_addNMS)
            {
                var centersToCornersData = new[]
                {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
                };
                var centersToCorners = Functional.Constant(new TensorShape(4, 4), centersToCornersData); // (4,4)
                var boxCorners = Functional.MatMul(slicedPositions[0, .., ..].Transpose(0, 1), centersToCorners); // (N,4)
                var indices = Functional.NMS(boxCorners, confidences[0, ..], _iouThreshold, _scoreThreshold); // (N)

                classIds = Functional.Gather(argMaxClasses, 1, indices.Unsqueeze(0)).Unsqueeze(1); // (1,1,N)
                scores = Functional.Gather(confidences, 1, indices.Unsqueeze(0)).Unsqueeze(1); // (1,1,N)
                coords = Functional.IndexSelect(slicedPositions, 2, indices); // (1,4,N)

            }
            else
            {
                coords = slicedPositions;
                classIds = argMaxClasses.Unsqueeze(1);
                scores = confidences.Unsqueeze(1);
            }
            var concatenated = Functional.Concat(new FunctionalTensor[] { coords, classIds.Float(), scores }, 1); // (1,6,N)

            model = graph.Compile(concatenated);
        }

    }

    public class YOLOCustomizationParameters
    {
        public bool AddClassificationHead { get; private set; }

        public bool QuantizeModel { get; private set; }
        public QuantizationType QuantizationType { get; private set; } = QuantizationType.Float16;

        public bool AddNMS { get; private set; }

        public float IoUThreshold;
        public float ScoreThreshold;
        
        public YOLOCustomizationParameters(bool addClassificationHead = true, YOLOQuantizationType yoloQuantizationType = YOLOQuantizationType.None, bool addNMS = false, float iouThreshold = 0.5f, float scoreThreshold = 0.5f)
        {
            AddClassificationHead = addClassificationHead;
            IoUThreshold = iouThreshold;
            ScoreThreshold = scoreThreshold;

            switch (yoloQuantizationType)
            {
                case YOLOQuantizationType.Float16:
                    QuantizeModel = true;
                    QuantizationType = QuantizationType.Float16;
                    break;
                case YOLOQuantizationType.Uint8:
                    QuantizeModel = true;
                    QuantizationType = QuantizationType.Uint8;
                    break;
                default:
                    QuantizeModel = false;
                    break;
            }

            AddNMS = addNMS;
        }
    }

    public class YOLOModel
    {
        public Model Model { get; private set; }

        public YOLOModel(Model yoloModel)
        {
            Model = yoloModel;
        }
    }

    public enum YOLOQuantizationType
    {
        None,
        Float16,
        Uint8
    }

    public enum YOLOCustomization
    {
        AddClassificationHead,
        None
    }

    public enum YOLONMSCustomization
    {
        AddNMS,
        None
    }
}