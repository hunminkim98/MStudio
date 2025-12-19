using System.Numerics;
using MStudio.Core.Parsers;
using Xunit;

namespace MStudio.Tests
{
    public class TrcFileParserTests
    {
        private readonly TrcFileParser _parser = new();

        [Fact]
        public void CanParse_TrcFile_ReturnsTrue()
        {
            Assert.True(_parser.CanParse("test.trc"));
            Assert.True(_parser.CanParse("TEST.TRC"));
        }

        [Fact]
        public void CanParse_OtherFile_ReturnsFalse()
        {
            Assert.False(_parser.CanParse("test.c3d"));
            Assert.False(_parser.CanParse("test.json"));
        }

        [Fact]
        public async Task ParseAsync_ValidTrcFile_ReturnsCorrectMetadata()
        {
            // 기존 Python 프로젝트의 테스트 파일 사용
            var testFilePath = Path.Combine(GetTestDataPath(), "test.trc");
            
            if (!File.Exists(testFilePath))
            {
                // 테스트 파일이 없으면 스킵
                return;
            }

            var result = await _parser.ParseAsync(testFilePath);

            Assert.NotNull(result);
            Assert.NotNull(result.Metadata);
            Assert.Equal(120.0f, result.Metadata.FrameRate);
            Assert.Equal(137, result.Metadata.TotalFrames);
            Assert.Equal(29, result.Metadata.MarkerNames.Count);
            Assert.Equal("m", result.Metadata.Units);
        }

        [Fact]
        public async Task ParseAsync_ValidTrcFile_ReturnsCorrectMarkerData()
        {
            var testFilePath = Path.Combine(GetTestDataPath(), "test.trc");
            
            if (!File.Exists(testFilePath))
            {
                return;
            }

            var result = await _parser.ParseAsync(testFilePath);

            Assert.NotNull(result.Markers);
            Assert.Equal(29, result.Markers.MarkerCount);
            Assert.Equal(137, result.Markers.FrameCount);

            // 첫 번째 프레임, 첫 번째 마커 (RHip) 데이터 검증
            var pos = result.Markers.GetPosition(0, 0);
            Assert.Equal(-0.2348508f, pos.X, 4);
            Assert.Equal(0.7315481f, pos.Y, 4);
            Assert.Equal(0.3067203f, pos.Z, 4);
        }

        private static string GetTestDataPath()
        {
            // MStudio.NET/tests/MStudio.Tests -> MStudio/tests
            var currentDir = Directory.GetCurrentDirectory();
            var mstudioRoot = Path.GetFullPath(Path.Combine(currentDir, "..", "..", "..", "..", ".."));
            return Path.Combine(mstudioRoot, "tests");
        }
    }
}
