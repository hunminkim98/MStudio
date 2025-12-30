using System.Collections.Generic;
using System.Numerics;

namespace MStudio.Core.Interfaces
{
    public enum AnalysisAxis
    {
        X, 
        Y, 
        Z
    }

    public interface IAnalysisService
    {
        /// <summary>
        /// Calculates the velocity vector using central difference.
        /// </summary>
        Vector3? CalculateVelocity(Vector3 posPrev, Vector3 posCurr, Vector3 posNext, float frameRate);
        
        /// <summary>
        /// Calculates the acceleration vector using central difference on velocity.
        /// </summary>
        Vector3? CalculateAcceleration(Vector3 velPrev, Vector3 velNext, float frameRate);
        
        /// <summary>
        /// Calculates the absolute angle of a vector (p1 -> p2) relative to a specific axis.
        /// </summary>
        float? CalculateAbsoluteAngle(Vector3 p1, Vector3 p2, AnalysisAxis axis);
        
        /// <summary>
        /// Calculates the relative angle formed by three points (p1 - vertex - p3).
        /// </summary>
        float? CalculateRelativeAngle(Vector3 p1, Vector3 vertex, Vector3 p3);
        
        /// <summary>
        /// Calculates the twist angle between two vectors (p2-p1) and (p4-p3) projected onto the Transverse Plane (XZ).
        /// </summary>
        float? CalculateTwistAngle(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4);

        /// <summary>
        /// Calculates points along an arc for visualization of the angle.
        /// </summary>
        List<Vector3>? CalculateArcPoints(Vector3 vertex, Vector3 p1, Vector3 p3, float radius = 0.05f, int segments = 20);
    }
}
