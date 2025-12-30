using System;
using System.Collections.Generic;
using System.Numerics;
using MStudio.Core.Interfaces;

namespace MStudio.Services.Implementations
{
    public class AnalysisService : IAnalysisService
    {
        public Vector3? CalculateVelocity(Vector3 posPrev, Vector3 posCurr, Vector3 posNext, float frameRate)
        {
            if (frameRate <= 0) return null;
            if (float.IsNaN(posPrev.X) || float.IsNaN(posNext.X)) return null;

            // Central difference: v(i) = (pos(i+1) - pos(i-1)) / (2 * dt)
            float dt = 1.0f / frameRate;
            return (posNext - posPrev) / (2.0f * dt);
        }

        public Vector3? CalculateAcceleration(Vector3 velPrev, Vector3 velNext, float frameRate)
        {
            if (frameRate <= 0) return null;
            if (float.IsNaN(velPrev.X) || float.IsNaN(velNext.X)) return null;

            // Central difference: a(i) = (vel(i+1) - vel(i-1)) / (2 * dt)
            float dt = 1.0f / frameRate;
            return (velNext - velPrev) / (2.0f * dt);
        }

        public float? CalculateAbsoluteAngle(Vector3 p1, Vector3 p2, AnalysisAxis axis)
        {
            if (float.IsNaN(p1.X) || float.IsNaN(p2.X)) return null;

            var vector = p2 - p1;
            if (vector.LengthSquared() == 0) return null;

            var normalizedVector = Vector3.Normalize(vector);
            
            // Reference axis vector
            Vector3 axisVector = axis switch
            {
                AnalysisAxis.X => Vector3.UnitX,
                AnalysisAxis.Y => Vector3.UnitY,
                AnalysisAxis.Z => Vector3.UnitZ, // Z-up is handled by caller context or standard Z axis
                _ => Vector3.UnitY
            };

            // Calculate angle with the reference axis using Dot product
            float dot = Vector3.Dot(normalizedVector, axisVector);
            
            // Clamp dot for Acos domain safety
            if (dot > 1.0f) dot = 1.0f;
            if (dot < -1.0f) dot = -1.0f;

            float angleRad = MathF.Acos(dot);
            return angleRad * (180.0f / MathF.PI);
        }

        public float? CalculateRelativeAngle(Vector3 p1, Vector3 vertex, Vector3 p3)
        {
            if (float.IsNaN(p1.X) || float.IsNaN(vertex.X) || float.IsNaN(p3.X)) return null;

            var v1 = p1 - vertex;
            var v3 = p3 - vertex;

            if (v1.LengthSquared() == 0 || v3.LengthSquared() == 0) return null;

            var v1Norm = Vector3.Normalize(v1);
            var v3Norm = Vector3.Normalize(v3);

            float dot = Vector3.Dot(v1Norm, v3Norm);
            
            // Clamp for safety
            if (dot > 1.0f) dot = 1.0f;
            if (dot < -1.0f) dot = -1.0f;

            float angleRad = MathF.Acos(dot);
            return angleRad * (180.0f / MathF.PI);
        }

        public List<Vector3>? CalculateArcPoints(Vector3 vertex, Vector3 p1, Vector3 p3, float radius = 0.05f, int segments = 20)
        {
            if (float.IsNaN(p1.X) || float.IsNaN(vertex.X) || float.IsNaN(p3.X)) return null;

            var v1 = p1 - vertex;
            var v3 = p3 - vertex;

            float len1 = v1.Length();
            float len3 = v3.Length();

            if (len1 == 0 || len3 == 0) return null;

            var v1Norm = v1 / len1;
            var v3Norm = v3 / len3;

            float dot = Vector3.Dot(v1Norm, v3Norm);
            if (dot > 1.0f) dot = 1.0f;
            if (dot < -1.0f) dot = -1.0f;

            float angleRad = MathF.Acos(dot);

            // If angle is effectively 0 or 180, usually no arc to draw (or line)
            if (angleRad < 0.001f || MathF.Abs(angleRad - MathF.PI) < 0.001f)
                return null;

            // Normal to the plane defined by v1 and v3
            var normal = Vector3.Cross(v1Norm, v3Norm);
            float normalLen = normal.Length();
            
            if (normalLen < 1e-6f) return null; // Collinear vectors

            var planeNormal = normal / normalLen;

            // Gram-Schmidt / Axis construction
            // axis1 = v1Norm
            // axis2 needs to be perpendicular to axis1 and in the plane
            var axis2 = Vector3.Cross(planeNormal, v1Norm);
            axis2 = Vector3.Normalize(axis2);

            var points = new List<Vector3>();
            for (int i = 0; i <= segments; i++)
            {
                float phi = angleRad * ((float)i / segments);
                
                // Point on circle = R * (axis1 * cos + axis2 * sin)
                var pointOnArc = vertex + radius * (v1Norm * MathF.Cos(phi) + axis2 * MathF.Sin(phi));
                points.Add(pointOnArc);
            }

            return points;
        }
        public float? CalculateTwistAngle(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4)
        {
            if (float.IsNaN(p1.X) || float.IsNaN(p2.X) || float.IsNaN(p3.X) || float.IsNaN(p4.X)) return null;

            // Project vectors onto XZ plane
            Vector2 v1Pro = new Vector2(p2.X - p1.X, p2.Z - p1.Z); // Reference Vector
            Vector2 v2Pro = new Vector2(p4.X - p3.X, p4.Z - p3.Z); // Target Vector

            if (v1Pro.LengthSquared() == 0 || v2Pro.LengthSquared() == 0) return null;

            // Angle relative to world X-axis
            float ang1 = MathF.Atan2(v1Pro.Y, v1Pro.X);
            float ang2 = MathF.Atan2(v2Pro.Y, v2Pro.X);

            float diff = ang2 - ang1;

            // Normalize to -180 ~ 180 degrees
            float diffDeg = diff * (180.0f / MathF.PI);

            while (diffDeg > 180.0f) diffDeg -= 360.0f;
            while (diffDeg < -180.0f) diffDeg += 360.0f;

            return diffDeg;
        }
    }
}
