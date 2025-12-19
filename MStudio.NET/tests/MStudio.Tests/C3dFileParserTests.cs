using System.IO;
using System.Threading.Tasks;
using MStudio.Core.Parsers;
using Xunit;

namespace MStudio.Tests
{
    public class C3dFileParserTests
    {
        private readonly C3dFileParser _parser = new();

        [Fact]
        public void CanParse_C3dFile_ReturnsTrue()
        {
            Assert.True(_parser.CanParse("test.c3d"));
        }

        [Fact]
        public async Task ParseAsync_ValidC3dFile_LoadsMetadata()
        {
            var testFilePath = Path.Combine(GetTestDataPath(), "test.c3d");
            if (!File.Exists(testFilePath)) return;

            var result = await _parser.ParseAsync(testFilePath);

            Assert.NotNull(result);
            Assert.NotNull(result.Metadata);
            // test.c3d header info based on common Gait2392 or similar
            Assert.True(result.Metadata.TotalFrames > 0);
            Assert.True(result.Metadata.MarkerNames.Count > 0);
        }

        private static string GetTestDataPath()
        {
            var currentDir = Directory.GetCurrentDirectory();
            var mstudioRoot = Path.GetFullPath(Path.Combine(currentDir, "..", "..", "..", "..", ".."));
            return Path.Combine(mstudioRoot, "tests");
        }
    }
}
