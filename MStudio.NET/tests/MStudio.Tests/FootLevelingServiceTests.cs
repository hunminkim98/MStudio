using MStudio.Core.Models;
using MStudio.Services.Implementations;

namespace MStudio.Tests
{
    public class FootLevelingServiceTests
    {
        private FootLevelingService CreateService() => new FootLevelingService();

        private MotionData CreateTestMotionData(string[] markerNames, int frameCount, float[,,] positions)
        {
            var markers = new MarkerDataContainer(markerNames.Length, frameCount);
            
            for (int frame = 0; frame < frameCount; frame++)
            {
                for (int marker = 0; marker < markerNames.Length; marker++)
                {
                    markers.SetPosition(marker, frame, 
                        positions[frame, marker, 0], 
                        positions[frame, marker, 1], 
                        positions[frame, marker, 2]);
                }
            }

            return new MotionData
            {
                Metadata = new MotionMetadata
                {
                    FilePath = "test.trc",
                    FrameRate = 30f,
                    TotalFrames = frameCount,
                    MarkerNames = markerNames
                },
                Markers = markers
            };
        }

        [Fact]
        public void GetFootMarkerIndices_WithFootMarkers_ReturnsCorrectIndices()
        {
            // Arrange
            var service = CreateService();
            var markerNames = new[] { "Nose", "LBigToe", "RHeel", "LAnkle", "RSmallToe" };

            // Act
            var indices = service.GetFootMarkerIndices(markerNames);

            // Assert
            Assert.Equal(3, indices.Count); // LBigToe, RHeel, RSmallToe
            Assert.Contains(1, indices); // LBigToe
            Assert.Contains(2, indices); // RHeel
            Assert.Contains(4, indices); // RSmallToe
        }

        [Fact]
        public void GetFootMarkerIndices_WithoutFootMarkers_ReturnsEmptyList()
        {
            // Arrange
            var service = CreateService();
            var markerNames = new[] { "Nose", "LAnkle", "RAnkle", "LKnee" };

            // Act
            var indices = service.GetFootMarkerIndices(markerNames);

            // Assert - No foot markers, empty list (no Ankle fallback)
            Assert.Empty(indices);
        }

        [Fact]
        public void HasFootMarkers_WithFootMarkers_ReturnsTrue()
        {
            var service = CreateService();
            var markerNames = new[] { "Nose", "LBigToe", "RAnkle" };
            Assert.True(service.HasFootMarkers(markerNames));
        }

        [Fact]
        public void HasFootMarkers_WithoutFootMarkers_ReturnsFalse()
        {
            var service = CreateService();
            var markerNames = new[] { "Nose", "LAnkle", "RAnkle" };
            Assert.False(service.HasFootMarkers(markerNames));
        }

        [Fact]
        public void ApplyFootLeveling_SubtractsOffsetFromAllMarkers()
        {
            // Arrange
            var service = CreateService();
            var markerNames = new[] { "Nose", "LBigToe", "RBigToe" };
            
            // 3 frames, foot markers at different heights
            // Frame 0: feet at Y=0.1
            // Frame 1: feet at Y=0.05 (lowest - ground frame)
            // Frame 2: feet at Y=0.15
            var positions = new float[3, 3, 3]
            {
                { { 0, 1.7f, 0 }, { 0, 0.1f, 0 }, { 0, 0.1f, 0 } },    // Frame 0
                { { 0, 1.65f, 0 }, { 0, 0.05f, 0 }, { 0, 0.05f, 0 } }, // Frame 1 (ground)
                { { 0, 1.75f, 0 }, { 0, 0.15f, 0 }, { 0, 0.15f, 0 } }  // Frame 2
            };
            
            var motionData = CreateTestMotionData(markerNames, 3, positions);

            // Act
            service.ApplyFootLeveling(motionData);

            // Assert - After applying, foot markers at ground frame should be at Y=0
            var footPos = motionData.Markers.GetPosition(1, 1); // LBigToe at frame 1
            Assert.True(Math.Abs(footPos.Y) < 0.001f, $"Expected Y≈0, got {footPos.Y}");
            
            // Nose should also be shifted by the same offset
            var nosePos = motionData.Markers.GetPosition(0, 1);
            Assert.True(Math.Abs(nosePos.Y - 1.6f) < 0.001f, $"Expected Y≈1.6, got {nosePos.Y}");
        }

        [Fact]
        public void ApplyFootLeveling_WithoutFootMarkers_ReturnsFalse()
        {
            // Arrange
            var service = CreateService();
            var positions = new float[1, 2, 3] { { { 0, 1.7f, 0 }, { 0, 0.1f, 0 } } };
            var motionData = CreateTestMotionData(new[] { "Nose", "LAnkle" }, 1, positions);

            // Act
            bool result = service.ApplyFootLeveling(motionData);

            // Assert
            Assert.False(result);
            Assert.False(service.IsApplied);
        }

        [Fact]
        public void ApplyFootLeveling_SetsIsAppliedToTrue()
        {
            // Arrange
            var service = CreateService();
            var positions = new float[1, 2, 3] { { { 0, 1.7f, 0 }, { 0, 0.1f, 0 } } };
            var motionData = CreateTestMotionData(new[] { "Nose", "LBigToe" }, 1, positions);

            // Act
            service.ApplyFootLeveling(motionData);

            // Assert
            Assert.True(service.IsApplied);
            Assert.True(Math.Abs(service.AppliedOffset - 0.1f) < 0.001f);
        }

        [Fact]
        public void UndoFootLeveling_RestoresOriginalPositions()
        {
            // Arrange
            var service = CreateService();
            var positions = new float[1, 2, 3] { { { 0, 1.7f, 0 }, { 0, 0.1f, 0 } } };
            var motionData = CreateTestMotionData(new[] { "Nose", "LBigToe" }, 1, positions);
            
            // Store original position
            var originalNoseY = motionData.Markers.GetPosition(0, 0).Y;

            // Apply and then Undo
            service.ApplyFootLeveling(motionData);
            service.UndoFootLeveling(motionData);

            // Assert - positions should be restored
            var restoredNoseY = motionData.Markers.GetPosition(0, 0).Y;
            Assert.True(Math.Abs(restoredNoseY - originalNoseY) < 0.001f, 
                $"Expected Y={originalNoseY}, got {restoredNoseY}");
        }

        [Fact]
        public void UndoFootLeveling_SetsIsAppliedToFalse()
        {
            // Arrange
            var service = CreateService();
            var positions = new float[1, 2, 3] { { { 0, 1.7f, 0 }, { 0, 0.1f, 0 } } };
            var motionData = CreateTestMotionData(new[] { "Nose", "LBigToe" }, 1, positions);

            // Apply and then Undo
            service.ApplyFootLeveling(motionData);
            service.UndoFootLeveling(motionData);

            // Assert
            Assert.False(service.IsApplied);
            Assert.Equal(0, service.AppliedOffset);
        }

        [Fact]
        public void ApplyFootLeveling_DoesNotReapplyIfAlreadyApplied()
        {
            // Arrange
            var service = CreateService();
            var positions = new float[1, 2, 3] { { { 0, 1.7f, 0 }, { 0, 0.1f, 0 } } };
            var motionData = CreateTestMotionData(new[] { "Nose", "LBigToe" }, 1, positions);

            // Apply twice
            service.ApplyFootLeveling(motionData);
            var firstOffset = service.AppliedOffset;
            var posAfterFirst = motionData.Markers.GetPosition(0, 0).Y;
            
            service.ApplyFootLeveling(motionData);
            var posAfterSecond = motionData.Markers.GetPosition(0, 0).Y;

            // Assert - second apply should have no effect
            Assert.Equal(firstOffset, service.AppliedOffset);
            Assert.Equal(posAfterFirst, posAfterSecond);
        }
    }
}
