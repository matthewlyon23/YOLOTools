using System.Collections;
#if UNITY_6000_2_OR_NEWER
using Unity.InferenceEngine;
#else
using Unity.Sentis;
#endif
using UnityEngine;

namespace YOLOTools.Inference
{
    public abstract class InferenceHandler<T>
    {
        protected Model _model;
        protected Worker _worker;

        public abstract Awaitable<Tensor<float>> Run(T input);

        public abstract IEnumerator RunWithLayerControl(T input);

        public abstract Tensor PeekOutput();

        public abstract void DisposeTensors();
    }
}