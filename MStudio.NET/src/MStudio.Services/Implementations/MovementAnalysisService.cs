using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MStudio.Core.Interfaces;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// Mock implementation of IMovementAnalysisService.
    /// Provides available analysis types and placeholder analysis results.
    /// </summary>
    public class MovementAnalysisService : IMovementAnalysisService
    {
        private static readonly IReadOnlyList<AnalysisTypeInfo> _analysisTypes = new List<AnalysisTypeInfo>
        {
            new(AnalysisType.Gait, "Gait", "Analyze walking patterns and gait parameters", "Icon_Gait"),
            new(AnalysisType.Running, "Running", "Analyze running mechanics and efficiency", "Icon_Running"),
            new(AnalysisType.Sprint, "Sprint", "Analyze sprint performance and acceleration", "Icon_Sprint"),
            new(AnalysisType.CounterMovementJump, "Counter Movement Jump", "Analyze CMJ height, power, and technique", "Icon_CMJ"),
            new(AnalysisType.BroadJump, "Broad Jump", "Analyze standing broad jump distance and mechanics", "Icon_BroadJump"),
            new(AnalysisType.ChangeOfDirection, "Change of Direction", "Analyze agility and direction change efficiency", "Icon_COD")
        };

        public IReadOnlyList<AnalysisTypeInfo> GetAvailableAnalysisTypes()
        {
            return _analysisTypes;
        }

        public Task<AnalysisResult> RunAnalysisAsync(AnalysisType type, MotionData? data)
        {
            // Mock implementation - returns placeholder result
            var typeInfo = _analysisTypes[(int)type];
            
            var result = new AnalysisResult
            {
                Type = type,
                AnalyzedAt = DateTime.Now,
                IsSuccess = true,
                Summary = $"{typeInfo.DisplayName} analysis selected. (Mock - Feature under development)"
            };

            return Task.FromResult(result);
        }
    }
}
