using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using OpenCvSharp;

namespace Deep3DStudio.Model.SfM
{
    /// <summary>
    /// Implements Bundle Adjustment using Levenberg-Marquardt optimization.
    /// Utilizes the Schur Complement to efficiently solve the sparse normal equations.
    /// </summary>
    public class BundleAdjustment
    {
        // Internal state representations optimized for MathNet
        public class BACamera
        {
            public int OriginalIndex;
            public Vector<double> Rotation; // Rodrigues vector (3)
            public Vector<double> Translation; // Translation vector (3)
            public Matrix<double> K; // Intrinsics (3x3)
            public bool IsFixed; // Fix first camera to gauge freedom
        }

        public class BAPoint
        {
            public int Id; // Map back to original
            public Vector<double> Position; // World position (3)
        }

        public class BAObservation
        {
            public int CameraIndex; // Index in our BACamera list
            public int PointIndex;  // Index in our BAPoint list
            public Vector<double> Observed; // (u, v)
        }

        private List<BACamera> _cameras;
        private List<BAPoint> _points;
        private List<BAObservation> _observations;

        public BundleAdjustment()
        {
            _cameras = new List<BACamera>();
            _points = new List<BAPoint>();
            _observations = new List<BAObservation>();
        }

        public void AddCamera(int originalIndex, Mat R, Mat t, Mat K, bool isFixed = false)
        {
            // Convert R (3x3) to Rodrigues (3x1)
            Mat rvec = new Mat();
            Cv2.Rodrigues(R, rvec);

            var cam = new BACamera
            {
                OriginalIndex = originalIndex,
                Rotation = Vector<double>.Build.DenseOfArray(new double[] { rvec.At<double>(0), rvec.At<double>(1), rvec.At<double>(2) }),
                Translation = Vector<double>.Build.DenseOfArray(new double[] { t.At<double>(0), t.At<double>(1), t.At<double>(2) }),
                K = Matrix<double>.Build.DenseOfArray(new double[,] {
                    { K.At<double>(0,0), K.At<double>(0,1), K.At<double>(0,2) },
                    { K.At<double>(1,0), K.At<double>(1,1), K.At<double>(1,2) },
                    { K.At<double>(2,0), K.At<double>(2,1), K.At<double>(2,2) }
                }),
                IsFixed = isFixed
            };
            _cameras.Add(cam);
        }

        public void AddPoint(int id, Point3d pt)
        {
            _points.Add(new BAPoint
            {
                Id = id,
                Position = Vector<double>.Build.DenseOfArray(new double[] { pt.X, pt.Y, pt.Z })
            });
        }

        public void AddObservation(int camIdxInList, int pointIdxInList, Point2d pt)
        {
            _observations.Add(new BAObservation
            {
                CameraIndex = camIdxInList,
                PointIndex = pointIdxInList,
                Observed = Vector<double>.Build.DenseOfArray(new double[] { pt.X, pt.Y })
            });
        }

        public void Optimize(int iterations = 10)
        {
            if (_cameras.Count == 0 || _points.Count == 0) return;

            Console.WriteLine($"Bundle Adjustment: {_cameras.Count} cams, {_points.Count} points, {_observations.Count} obs.");

            // Levenberg-Marquardt parameters
            double lambda = 1e-3;
            double initialError = ComputeTotalReprojectionError();
            Console.WriteLine($"Initial MSE: {Math.Sqrt(initialError / _observations.Count):F4} px");

            for (int iter = 0; iter < iterations; iter++)
            {
                // 1. Build Linear System (J^T J delta = -J^T r)
                // We use the Schur Complement method to solve for Camera updates first, then Point updates.
                // Or for simplicity in this implementation (to ensure stability), we might try dense if small,
                // but let's try a simplified iterative update if implementing full Schur is too verbose.
                //
                // Actually, let's implement the standard structure:
                // H = [ H_cc  H_cp ]
                //     [ H_pc  H_pp ]
                // H_pp is block diagonal (3x3 blocks). Inverting it is easy.
                //
                // Reduced Camera System: (H_cc - H_cp * inv(H_pp) * H_pc) * delta_c = -b_c + H_cp * inv(H_pp) * b_p

                if (!SolveLMStep(ref lambda))
                {
                    // If lambda gets too huge or step fails repeatedly
                    if (lambda > 1e10) break;
                }

                double currentError = ComputeTotalReprojectionError();
                // Console.WriteLine($"Iter {iter}: MSE = {Math.Sqrt(currentError / _observations.Count):F4} px");

                if (currentError < 1e-6) break;
            }

            Console.WriteLine($"Final MSE: {Math.Sqrt(ComputeTotalReprojectionError() / _observations.Count):F4} px");
        }

        // Apply updated parameters back to OpenCV Mats
        public void UpdateResults(List<OpenCvSharp.Mat> rotations, List<OpenCvSharp.Mat> translations, Dictionary<int, Point3d> points)
        {
            // Update Cameras
            for(int i=0; i<_cameras.Count; i++)
            {
                // Convert Rodrigues vector back to Rotation Matrix
                var rVecArr = _cameras[i].Rotation.ToArray();
                using var rVecMat = new Mat(3, 1, MatType.CV_64F);
                rVecMat.Set(0, 0, rVecArr[0]);
                rVecMat.Set(1, 0, rVecArr[1]);
                rVecMat.Set(2, 0, rVecArr[2]);

                Cv2.Rodrigues(rVecMat, rotations[i]); // Writes into the provided Mat (ViewInfo.R)

                // Update Translation
                var tArr = _cameras[i].Translation.ToArray();
                translations[i].Set(0, 0, tArr[0]);
                translations[i].Set(1, 0, tArr[1]);
                translations[i].Set(2, 0, tArr[2]);
            }

            // Update Points
            foreach(var p in _points)
            {
                if(points.ContainsKey(p.Id))
                {
                    points[p.Id] = new Point3d(p.Position[0], p.Position[1], p.Position[2]);
                }
            }
        }

        private double ComputeTotalReprojectionError()
        {
            double errorSq = 0;
            foreach (var obs in _observations)
            {
                var cam = _cameras[obs.CameraIndex];
                var pt = _points[obs.PointIndex];
                var proj = Project(cam, pt.Position);
                var diff = obs.Observed - proj;
                errorSq += diff.DotProduct(diff);
            }
            return errorSq;
        }

        private Vector<double> Project(BACamera cam, Vector<double> pointWorld)
        {
            // Rodrigues to Rotation Matrix
            // We implement Rodrigues formula manually for derivatives later, but here just value.
            // Or use Cv2 for projection value to be safe? No, mixing types is slow.
            // Let's implement simple projection logic.

            // 1. Rotate
            // R = I + sin(theta) K + (1-cos(theta)) K^2
            var rVec = cam.Rotation;
            double theta = rVec.L2Norm();
            Vector<double> pCam;

            if (theta < 1e-8)
            {
                pCam = pointWorld + cam.Translation;
            }
            else
            {
                var k = rVec / theta;
                var P = pointWorld;
                // Rodrigues rotation of vector P: P_rot = P cos(th) + (k x P) sin(th) + k(k.P)(1 - cos(th))
                var kCrossP = CrossProduct(k, P);
                var kDotP = k.DotProduct(P);
                var P_rot = P * Math.Cos(theta) + kCrossP * Math.Sin(theta) + k * (kDotP * (1.0 - Math.Cos(theta)));
                pCam = P_rot + cam.Translation;
            }

            // 2. Project
            if (pCam[2] <= 1e-6) return Vector<double>.Build.Dense(2, -10000); // Behind camera

            double x = pCam[0] / pCam[2];
            double y = pCam[1] / pCam[2];

            // 3. Intrinsics
            double fx = cam.K[0, 0];
            double fy = cam.K[1, 1];
            double cx = cam.K[0, 2];
            double cy = cam.K[1, 2];

            return Vector<double>.Build.DenseOfArray(new double[] { fx * x + cx, fy * y + cy });
        }

        private Vector<double> CrossProduct(Vector<double> a, Vector<double> b)
        {
            return Vector<double>.Build.DenseOfArray(new double[] {
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            });
        }

        private bool SolveLMStep(ref double lambda)
        {
            // Build Hessian (approx J^T J) and Gradient (J^T r)
            // Due to size, we build block components for Schur Complement

            int numCams = _cameras.Count;
            int numPoints = _points.Count;

            // H_cc is block diagonal with 6x6 blocks
            var H_cc = new Matrix<double>[numCams];
            var b_c = new Vector<double>[numCams];

            // H_pp is block diagonal with 3x3 blocks
            var H_pp = new Matrix<double>[numPoints];
            var b_p = new Vector<double>[numPoints];

            for (int i = 0; i < numCams; i++) { H_cc[i] = Matrix<double>.Build.Dense(6, 6); b_c[i] = Vector<double>.Build.Dense(6); }
            for (int i = 0; i < numPoints; i++) { H_pp[i] = Matrix<double>.Build.Dense(3, 3); b_p[i] = Vector<double>.Build.Dense(3); }

            // Off-diagonal H_cp blocks (sparse storage)
            // Store as Dictionary mapping (camIdx, ptIdx) -> 6x3 Matrix
            // Actually, we can accumulate directly into the Reduced Camera System if we iterate carefully.
            // But let's follow the standard construction.
            var H_cp = new Dictionary<(int, int), Matrix<double>>();

            double currentTotalError = 0;

            foreach (var obs in _observations)
            {
                int cIdx = obs.CameraIndex;
                int pIdx = obs.PointIndex;
                var cam = _cameras[cIdx];
                var pt = _points[pIdx];

                // Compute Residual
                var proj = Project(cam, pt.Position);
                var r = obs.Observed - proj; // residual = obs - proj
                currentTotalError += r.DotProduct(r);

                // Compute Jacobians
                // J_c (2x6), J_p (2x3)
                ComputeJacobians(cam, pt.Position, out Matrix<double> J_c, out Matrix<double> J_p);

                // Accumulate H and b
                // H_cc += J_c^T * J_c
                var JcT = J_c.Transpose();
                var JpT = J_p.Transpose();

                H_cc[cIdx] += JcT * J_c;
                H_pp[pIdx] += JpT * J_p;

                // H_cp = J_c^T * J_p
                H_cp[(cIdx, pIdx)] = JcT * J_p;

                // b_c += J_c^T * r
                b_c[cIdx] += JcT * r;
                b_p[pIdx] += JpT * r;
            }

            // Apply Damping (Augment Diagonal)
            for (int i = 0; i < numCams; i++)
            {
                for (int j = 0; j < 6; j++) H_cc[i][j, j] *= (1.0 + lambda);
            }
            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < 3; j++) H_pp[i][j, j] *= (1.0 + lambda);
            }

            // Fix first camera (gauge freedom) or any fixed cameras
            for (int i = 0; i < numCams; i++)
            {
                if (_cameras[i].IsFixed)
                {
                    // Zero out rows/cols or set extremely high prior?
                    // Easiest: Set H_cc diagonal to huge, b_c to 0
                    H_cc[i].Clear();
                    for(int k=0; k<6; k++) H_cc[i][k, k] = 1e15;
                    b_c[i].Clear();
                }
            }

            // Schur Complement to solve for delta_c
            // S = H_cc - H_cp * inv(H_pp) * H_pc
            // rhs = b_c - H_cp * inv(H_pp) * b_p

            // Since H_cc is block diagonal, we can build S as a dense matrix of size (6*Nc)x(6*Nc)
            // or sparse. Typically it's dense-ish block structure.
            // Let's map global indices.

            int dimC = 6 * numCams;
            var S = Matrix<double>.Build.Dense(dimC, dimC);
            var rhs = Vector<double>.Build.Dense(dimC);

            // Fill S with H_cc and rhs with b_c initially
            for (int i = 0; i < numCams; i++)
            {
                S.SetSubMatrix(i * 6, 0, i * 6, 0, H_cc[i]);
                rhs.SetSubVector(i * 6, 6, b_c[i]);
            }

            // Precompute inv(H_pp) and process off-diagonals
            // For each point, compute contribution to S

            for (int j = 0; j < numPoints; j++)
            {
                // Invert H_pp[j] (3x3)
                // If singular (point not constrained), handle gracefully (add epsilon)
                try
                {
                    var HppInv = H_pp[j].Inverse();
                    var bp = b_p[j];
                    var HppInv_bp = HppInv * bp; // 3x1

                    // Find all cameras observing this point
                    var obsCams = _observations.Where(o => o.PointIndex == j).Select(o => o.CameraIndex).Distinct().ToList();

                    for (int k = 0; k < obsCams.Count; k++)
                    {
                        int c1 = obsCams[k];
                        var H_c1p = H_cp[(c1, j)]; // 6x3
                        var term1 = H_c1p * HppInv_bp; // 6x1

                        // Update rhs block for c1
                        var currentRhs = rhs.SubVector(c1 * 6, 6);
                        rhs.SetSubVector(c1 * 6, 6, currentRhs - term1);

                        for (int l = 0; l < obsCams.Count; l++)
                        {
                            int c2 = obsCams[l];
                            var H_c2p = H_cp[(c2, j)]; // 6x3
                            var H_pc2 = H_c2p.Transpose(); // 3x6

                            // W = H_c1p * inv(H_pp) * H_pc2
                            var W = H_c1p * HppInv * H_pc2; // 6x6

                            // S_c1c2 -= W
                            // Indices in S
                            int rowStart = c1 * 6;
                            int colStart = c2 * 6;

                            // Subtract W
                            // S is symmetric, but we fill fully for ease
                            // S[row, col] -= W
                            // MathNet SetSubMatrix adds? No, replaces. Need Get, Sub, Set.
                            var block = S.SubMatrix(rowStart, 6, colStart, 6);
                            S.SetSubMatrix(rowStart, colStart, block - W);
                        }
                    }
                }
                catch (Exception) {
                    // Singular point, ignore or fix?
                    // Usually happens if point observed by < 2 cams with bad geometry
                }
            }

            // Solve S * delta_c = rhs
            Vector<double> delta_c;
            try
            {
                delta_c = S.Solve(rhs);
            }
            catch
            {
                lambda *= 10;
                return false;
            }

            // Back-substitute for delta_p
            var delta_p = new Vector<double>[numPoints];
            for (int j = 0; j < numPoints; j++)
            {
                // delta_p = inv(H_pp) * (b_p - H_pc * delta_c)
                // H_pc is sum over cameras

                var sumHpcDc = Vector<double>.Build.Dense(3);
                var obsCams = _observations.Where(o => o.PointIndex == j).Select(o => o.CameraIndex).Distinct().ToList();

                foreach (int c in obsCams)
                {
                    var H_pc = H_cp[(c, j)].Transpose(); // 3x6
                    var dc = delta_c.SubVector(c * 6, 6);
                    sumHpcDc += H_pc * dc;
                }

                try {
                    delta_p[j] = H_pp[j].Inverse() * (b_p[j] - sumHpcDc);
                } catch { delta_p[j] = Vector<double>.Build.Dense(3); }
            }

            // Evaluate New State
            // Clone state
            var oldCams = _cameras.Select(c => new { R = c.Rotation.Clone(), T = c.Translation.Clone() }).ToList();
            var oldPoints = _points.Select(p => p.Position.Clone()).ToList();

            // Apply updates
            // Update rule for rotation: R_new = exp(delta_phi) * R_old
            // We use the "local parameterization" on the tangent space.

            for (int i = 0; i < numCams; i++)
            {
                if (_cameras[i].IsFixed) continue;

                var delta_rot = delta_c.SubVector(i * 6, 3);
                var delta_trans = delta_c.SubVector(i * 6 + 3, 3);

                // Update Translation
                _cameras[i].Translation += delta_trans;

                // Update Rotation using Lie Algebra
                // 1. Convert current Rodrigues to R_old
                var rVecArr = _cameras[i].Rotation.ToArray();
                using var rVecMat = new Mat(3, 1, MatType.CV_64F);
                rVecMat.Set(0, 0, rVecArr[0]);
                rVecMat.Set(1, 0, rVecArr[1]);
                rVecMat.Set(2, 0, rVecArr[2]);
                using var R_old = new Mat();
                Cv2.Rodrigues(rVecMat, R_old);

                // 2. Convert delta_rot to Delta_R matrix (exp map)
                // Rodrigues function also does exp map for 3x1 vector
                using var deltaVecMat = new Mat(3, 1, MatType.CV_64F);
                deltaVecMat.Set(0, 0, delta_rot[0]);
                deltaVecMat.Set(1, 0, delta_rot[1]);
                deltaVecMat.Set(2, 0, delta_rot[2]);
                using var R_delta = new Mat();
                Cv2.Rodrigues(deltaVecMat, R_delta);

                // 3. R_new = R_delta * R_old (Left multiply because Jacobian derived for left perturbation)
                // Note: d(R*P)/d(w) ~ -[R*P]x for Left perturbation (P_c_new = (I+[w]x)*P_c)
                // This corresponds to R_new = (I+[w]x) * R_old ~ exp(w) * R_old
                using var R_new = R_delta * R_old;

                // 4. Convert back to Rodrigues
                using var rVecNew = new Mat();
                Cv2.Rodrigues(R_new, rVecNew);

                _cameras[i].Rotation[0] = rVecNew.At<double>(0);
                _cameras[i].Rotation[1] = rVecNew.At<double>(1);
                _cameras[i].Rotation[2] = rVecNew.At<double>(2);
            }

            for (int j = 0; j < numPoints; j++)
            {
                _points[j].Position += delta_p[j];
            }

            // Check Error
            double newError = ComputeTotalReprojectionError();

            if (newError < currentTotalError)
            {
                // Accept
                lambda /= 10.0;
                return true;
            }
            else
            {
                // Reject - Restore
                for (int i = 0; i < numCams; i++)
                {
                    _cameras[i].Rotation = oldCams[i].R;
                    _cameras[i].Translation = oldCams[i].T;
                }
                for (int j = 0; j < numPoints; j++)
                {
                    _points[j].Position = oldPoints[j];
                }
                lambda *= 10.0;
                return false;
            }
        }

        private void ComputeJacobians(BACamera cam, Vector<double> Pw, out Matrix<double> J_c, out Matrix<double> J_p)
        {
            // Numerical or Analytical? Analytical is better.
            // P_c = R * P_w + t
            // p_uv = Project(P_c)

            // J_c: [d(p)/d(r) | d(p)/d(t)] (2x6)
            // J_p: d(p)/d(Pw) (2x3)

            // 1. Transform Pw to Pc
            // We need R matrix from Rodrigues vector
            var rVec = cam.Rotation;
            double theta = rVec.L2Norm();
            Matrix<double> R;

            // Rodrigues Formula for Matrix
            // K = [0 -z y; z 0 -x; -y x 0]
            var K_skew = Matrix<double>.Build.Dense(3, 3);
            K_skew[0, 1] = -rVec[2]; K_skew[0, 2] = rVec[1];
            K_skew[1, 0] = rVec[2];  K_skew[1, 2] = -rVec[0];
            K_skew[2, 0] = -rVec[1]; K_skew[2, 1] = rVec[0];

            // Normalize axis
            if (theta > 1e-8)
            {
                var axis = rVec / theta;
                var K_axis = Matrix<double>.Build.Dense(3, 3);
                K_axis[0, 1] = -axis[2]; K_axis[0, 2] = axis[1];
                K_axis[1, 0] = axis[2];  K_axis[1, 2] = -axis[0];
                K_axis[2, 0] = -axis[1]; K_axis[2, 1] = axis[0];

                R = Matrix<double>.Build.DenseIdentity(3) +
                    Math.Sin(theta) * K_axis +
                    (1 - Math.Cos(theta)) * (K_axis * K_axis);
            }
            else
            {
                R = Matrix<double>.Build.DenseIdentity(3) + K_skew;
            }

            var RPw = R * Pw; // Rotated point without translation
            var Pc = RPw + cam.Translation; // Camera coordinates
            double x = Pc[0], y = Pc[1], z = Pc[2];
            double z2 = z * z;

            // 2. d(p_uv) / d(P_c) (2x3)
            // u = fx * x/z + cx
            // v = fy * y/z + cy
            // du/dx = fx/z, du/dy = 0, du/dz = -fx*x/z^2

            double fx = cam.K[0, 0];
            double fy = cam.K[1, 1];

            var J_proj = Matrix<double>.Build.Dense(2, 3);
            J_proj[0, 0] = fx / z;
            J_proj[0, 2] = -fx * x / z2;
            J_proj[1, 1] = fy / z;
            J_proj[1, 2] = -fy * y / z2;

            // 3. d(P_c) / d(P_w) = R (3x3)
            J_p = J_proj * R;

            // 4. d(P_c) / d(t) = I (3x3)
            var J_t = J_proj * Matrix<double>.Build.DenseIdentity(3); // 2x3

            // 5. d(P_c) / d(r) (3x3)
            // Left Perturbation: R_new = exp(delta) * R
            // P_c_new = exp(delta) * R * Pw + t ~= (I + [delta]x) * R * Pw + t
            // P_c_new = R * Pw + t + [delta]x * (R * Pw)
            // d(P_c) / d(delta) = -[R * Pw]x

            double rx = RPw[0], ry = RPw[1], rz = RPw[2];
            var RPw_skew = Matrix<double>.Build.Dense(3, 3);
            RPw_skew[0, 1] = -rz; RPw_skew[0, 2] = ry;
            RPw_skew[1, 0] = rz;  RPw_skew[1, 2] = -rx;
            RPw_skew[2, 0] = -ry; RPw_skew[2, 1] = rx;

            // J_r = J_proj * (-RPw_skew)
            var J_r = J_proj * (-RPw_skew); // 2x3

            // Combine J_c = [J_r | J_t] (2x6)
            J_c = Matrix<double>.Build.Dense(2, 6);
            J_c.SetSubMatrix(0, 0, J_r);
            J_c.SetSubMatrix(0, 3, J_t);
        }
    }
}
