using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Core.Writers
{
    /// <summary>
    /// Writes motion data to C3D (Coordinate 3D) file format.
    /// Ported from Python MStudio dataSaver.py using c3d library concepts.
    /// 
    /// C3D file structure:
    /// - Header section (512 bytes, Block 1)
    /// - Parameter section (variable size, starting Block 2)
    /// - Data section (3D point data)
    /// 
    /// Scale Factor 의미:
    /// - 양수: 정수 형식 (Integer format) - 좌표를 int16으로 저장하고 ScaleFactor를 곱해서 실제 값 계산
    /// - 음수: 부동소수점 형식 (Floating point format) - 좌표를 float로 직접 저장 (부호만 의미있음)
    /// </summary>
    public static class C3dFileWriter
    {
        private const int BlockSize = 512;
        
        // 부동소수점 형식 사용 (음수 = float format, 절대값은 의미없음)
        private const float ScaleFactorFloatMode = -1.0f;
        
        /// <summary>
        /// Writes motion data to a C3D file asynchronously.
        /// Converts coordinates from meters to millimeters.
        /// </summary>
        /// <param name="filePath">Path to save the C3D file</param>
        /// <param name="motionData">Motion data to save</param>
        public static async Task WriteAsync(string filePath, MotionData motionData)
        {
            await Task.Run(() => WriteSync(filePath, motionData));
        }
        
        private static void WriteSync(string filePath, MotionData motionData)
        {
            var metadata = motionData.Metadata;
            var markers = motionData.Markers;
            var markerNames = metadata.MarkerNames;
            int numMarkers = markerNames.Count;
            int numFrames = metadata.TotalFrames;
            float frameRate = metadata.FrameRate;
            
            // 파라미터 섹션을 먼저 생성하여 실제 크기 계산
            byte[] parameterSection = BuildParameterSection(markerNames, frameRate, numFrames);
            
            // 파라미터 섹션이 차지하는 블록 수 계산 (512 바이트 단위)
            int parameterBlocks = (parameterSection.Length + BlockSize - 1) / BlockSize;
            
            // 데이터 시작 블록 = 헤더(1) + 파라미터 블록 수 + 1
            int dataStartBlock = 1 + parameterBlocks + 1;
            
            using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);
            
            // Block 1: Header (512 bytes)
            WriteHeader(writer, numMarkers, numFrames, frameRate, dataStartBlock);
            
            // Block 2~N: Parameter Section (padded to 512-byte blocks)
            writer.Write(parameterSection);
            
            // Pad to block boundary
            int paddingNeeded = (parameterBlocks * BlockSize) - parameterSection.Length;
            if (paddingNeeded > 0)
            {
                writer.Write(new byte[paddingNeeded]);
            }
            
            // Data Section
            WriteDataSection(writer, markers, numMarkers, numFrames);
        }
        
        private static void WriteHeader(BinaryWriter writer, int numMarkers, int numFrames, float frameRate, int dataStartBlock)
        {
            // Byte 1: Parameter block number (always 2 for modern C3D)
            writer.Write((byte)2);
            
            // Byte 2: C3D identifier (0x50 = 80)
            writer.Write((byte)0x50);
            
            // Bytes 3-4: Number of 3D points (markers) - ushort로 최대 65535개 지원
            writer.Write((ushort)numMarkers);
            
            // Bytes 5-6: Number of analog measurements per 3D frame
            writer.Write((ushort)0);
            
            // Bytes 7-8: First frame number
            writer.Write((ushort)1);
            
            // Bytes 9-10: Last frame number (ushort max = 65535)
            writer.Write((ushort)Math.Min(numFrames, ushort.MaxValue));
            
            // Bytes 11-12: Maximum interpolation gap
            writer.Write((ushort)0);
            
            // Bytes 13-16: Scale factor (negative = floating point format)
            writer.Write(ScaleFactorFloatMode);
            
            // Bytes 17-18: Data start block number
            writer.Write((ushort)dataStartBlock);
            
            // Bytes 19-20: Analog samples per frame
            writer.Write((ushort)0);
            
            // Bytes 21-24: Frame rate
            writer.Write(frameRate);
            
            // Fill rest of header block with zeros
            int headerBytesWritten = 24;
            writer.Write(new byte[BlockSize - headerBytesWritten]);
        }
        
        /// <summary>
        /// 파라미터 섹션을 빌드하고 바이트 배열로 반환합니다.
        /// 마커 수에 따라 크기가 동적으로 결정됩니다.
        /// </summary>
        private static byte[] BuildParameterSection(IReadOnlyList<string> markerNames, float frameRate, int numFrames)
        {
            using var paramStream = new MemoryStream();
            using var paramWriter = new BinaryWriter(paramStream);
            
            // Parameter section header (4 bytes)
            paramWriter.Write((byte)1);  // Reserved
            paramWriter.Write((byte)1);  // Reserved
            paramWriter.Write((byte)0);  // Number of parameter blocks (placeholder, updated later)
            paramWriter.Write((byte)0x54); // Processor type (Intel = 0x54)
            
            // Write POINT group
            WriteGroup(paramWriter, 1, "POINT", "3D point parameters");
            
            // POINT:LABELS parameter - 마커 이름 (가장 큰 부분)
            WriteStringArrayParameter(paramWriter, 1, "LABELS", markerNames);
            
            // POINT:RATE parameter
            WriteFloatParameter(paramWriter, 1, "RATE", frameRate);
            
            // POINT:FRAMES parameter
            WriteIntParameter(paramWriter, 1, "FRAMES", numFrames);
            
            // POINT:USED parameter
            WriteIntParameter(paramWriter, 1, "USED", markerNames.Count);
            
            // POINT:SCALE parameter
            WriteFloatParameter(paramWriter, 1, "SCALE", ScaleFactorFloatMode);
            
            // 파라미터 블록 수 계산 및 업데이트
            int totalSize = (int)paramStream.Length;
            int numBlocks = (totalSize + BlockSize - 1) / BlockSize;
            
            // 바이트 2 위치에 블록 수 업데이트
            byte[] result = paramStream.ToArray();
            result[2] = (byte)numBlocks;
            
            return result;
        }
        
        private static void WriteGroup(BinaryWriter writer, int groupId, string name, string description)
        {
            // Group name length (negative for group)
            writer.Write((sbyte)(-name.Length));
            // Group ID
            writer.Write((sbyte)(-groupId));
            // Group name
            writer.Write(Encoding.ASCII.GetBytes(name.ToUpper().PadRight(name.Length)));
            // Offset to next item (placeholder)
            writer.Write((short)(3 + description.Length));
            // Description length
            writer.Write((byte)description.Length);
            // Description
            writer.Write(Encoding.ASCII.GetBytes(description));
        }
        
        private static void WriteStringArrayParameter(BinaryWriter writer, int groupId, string name, IReadOnlyList<string> values)
        {
            // Parameter name length
            writer.Write((sbyte)name.Length);
            // Group ID
            writer.Write((sbyte)groupId);
            // Parameter name
            writer.Write(Encoding.ASCII.GetBytes(name.ToUpper()));
            
            // 마커 이름 최대 길이 계산 (최소 4, 최대 32)
            int maxLen = 4;
            foreach (var v in values)
            {
                maxLen = Math.Max(maxLen, v.Length);
            }
            maxLen = Math.Min(maxLen, 32); // C3D spec max
            
            // Calculate data size
            int dataSize = 2 + 2 + (values.Count * maxLen); // dims + dimensions + data
            
            // Offset to next
            writer.Write((short)(2 + dataSize));
            
            // Data type (-1 = char array)
            writer.Write((sbyte)(-1));
            
            // Number of dimensions
            writer.Write((byte)2);
            
            // Dimension 1: string length
            writer.Write((byte)maxLen);
            
            // Dimension 2: number of strings
            writer.Write((byte)values.Count);
            
            // String data
            foreach (var value in values)
            {
                var padded = value.PadRight(maxLen).Substring(0, maxLen);
                writer.Write(Encoding.ASCII.GetBytes(padded));
            }
        }
        
        private static void WriteFloatParameter(BinaryWriter writer, int groupId, string name, float value)
        {
            writer.Write((sbyte)name.Length);
            writer.Write((sbyte)groupId);
            writer.Write(Encoding.ASCII.GetBytes(name.ToUpper()));
            writer.Write((short)(2 + 4)); // offset: type + dims + data
            writer.Write((sbyte)4); // float type
            writer.Write((byte)0); // 0 dimensions (scalar)
            writer.Write(value);
        }
        
        private static void WriteIntParameter(BinaryWriter writer, int groupId, string name, int value)
        {
            writer.Write((sbyte)name.Length);
            writer.Write((sbyte)groupId);
            writer.Write(Encoding.ASCII.GetBytes(name.ToUpper()));
            writer.Write((short)(2 + 2)); // offset: type + dims + data
            writer.Write((sbyte)2); // int16 type
            writer.Write((byte)0); // 0 dimensions (scalar)
            writer.Write((short)value);
        }
        
        private static void WriteDataSection(BinaryWriter writer, MarkerDataContainer markers, int numMarkers, int numFrames)
        {
            // 부동소수점 형식: 각 포인트당 4개의 float (X, Y, Z, Residual)
            
            for (int frame = 0; frame < numFrames; frame++)
            {
                for (int m = 0; m < numMarkers; m++)
                {
                    var pos = markers.GetPosition(m, frame);
                    
                    // Convert from meters to millimeters
                    float x = pos.X * 1000.0f;
                    float y = pos.Y * 1000.0f;
                    float z = pos.Z * 1000.0f;
                    
                    // Handle missing data (NaN or zero position)
                    bool isMissing = float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0);
                    if (isMissing)
                    {
                        x = 0;
                        y = 0;
                        z = 0;
                    }
                    
                    writer.Write(x);
                    writer.Write(y);
                    writer.Write(z);
                    
                    // Residual + camera mask (combined as float in floating-point format)
                    // Residual = -1.0 for invalid points, 0.0 for valid
                    float residual = isMissing ? -1.0f : 0.0f;
                    writer.Write(residual);
                }
            }
        }
    }
}
