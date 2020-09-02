using HEWrapper;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class LLRollLayer : BaseLayer
    {
        public int Maps { get; set; } = 1;

        public double WeightsScale { get; set; } = 1.0;

        public override double GetOutputScale()
        {
            return WeightsScale * Source.GetOutputScale();
        }
        IVector[] weightsVector = null;
        IVector biasVector = null;

        public Vector<double> HotIndices { get; set; } = null;

        public override void Dispose()
        {
            if (weightsVector != null)
                foreach (var w in weightsVector)
                    if (w != null) w.Dispose();
            if (biasVector != null)
                biasVector.Dispose();
            weightsVector = null;
        }

        public override void Prepare()
        {
            if (!layerPrepared)
            {
                int dim = Source.OutputDimension();
                weightsVector = new IVector[dim];
                ParallelProcessInEnv(dim, (env, taskIndex, mapIndex) =>
                {
                    var v = Vector<double>.Build.Dense(OutputDimension()) + 1 + mapIndex;
                    weightsVector[mapIndex] = Factory.GetPlainVector(v, EVectorFormat.dense, WeightsScale);
                });
                var b = Vector<double>.Build.Dense(OutputDimension()) -1;
                biasVector = Factory.GetPlainVector(b, EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
            }
        }
        public override IMatrix Apply(IMatrix m)
        {
            if (m.ColumnCount > 1) throw new Exception("Expecting only one column");
            IVector[] rolls = new IVector[Maps];
            var vector = m.GetColumn(0);
            ParallelProcessInEnv(Maps, (env, task, k) =>
            {
                using (var mul = vector.PointwiseMultiply(weightsVector[k], env))
                    rolls[k] = mul.Rotate(k, env);
            });
            ProcessInEnv(env =>
            {
                for (int i = 1; i < Maps; i++)
                {
                    rolls[0].Add(rolls[i], env);
                    rolls[i].Dispose();
                }
            });
            return Factory.GetMatrix(new IVector[] { rolls[0] }, EMatrixFormat.ColumnMajor, CopyVectors: false);

        }
    }
}
