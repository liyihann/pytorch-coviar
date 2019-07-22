// CoViAR python data loader.
// Part of this implementation is modified from the tutorial at
// https://blog.csdn.net/leixiaohua1020/article/details/50618190
// and FFmpeg extract_mv example.


#include <Python.h>
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdio.h>
#include <omp.h>

#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>

#define FF_INPUT_BUFFER_PADDING_SIZE 32
#define MV 1
#define RESIDUAL 2

static const char *filename = NULL;


static PyObject *CoviarError;



void create_and_load_bgr(AVFrame *pFrame, AVFrame *pFrameBGR, uint8_t *buffer,
  PyArrayObject ** arr, int cur_pos, int pos_target) {

  // This function creates and loads a brg image.
  // :param pFrame: pFrame is an AVFrame containing a decompressed frame. Here, AVFrame is a structure describing decoded raw audio or video data.
  // :param pFrameBGR: pFrame contains some raw frame. pFrameBGR has been processed from pFrame (seems to be reshaping). Later, pFrameBGR will be copied to pyArray for output.
  // :param buffer: An unint8* pointer to the image buffer.
  // :param arr: allocated Zero-Value pyArray.
  // :param cur_pos: Current position.
  // :param pos_target: Desired position.

  // This function returns the number of bytes required to store a frame with spatial (pFrame->width, pFrame->height)
  // and pixel format AV_PIX_FMT_BGR24.
  int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
  // This line allocates a block of numBytes * sizeof(uint8_t) bytes with alignment suitable for all memory accesses (including vectors if available on the CPU).
  buffer = (uint8_t*) av_malloc(numBytes * sizeof(uint8_t));
  // AVPicture is a structure storing image datas and metadatas.
  // Setup the picture fields based on the specified image parameters and the provided image data buffer.
  avpicture_fill((AVPicture*) pFrameBGR, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);

  // Pointer to scaling context.
  struct SwsContext *img_convert_ctx;
  // Allocate a new scaling context.
  // Allocate and return an SwsContext.
  // :param srcW: pFrame->width
  // :param srcH: pFrame->height
  // :param srcFormat: AV_PIX_FMT_YUV420P
  // :param dstW: pFrame->width
  // :param dstH: pFrame->height
  // :param dstFormat: AV_PIX_FMT_BGR24
  img_convert_ctx = sws_getCachedContext(NULL,
      pFrame->width, pFrame->height, AV_PIX_FMT_YUV420P,
      pFrame->width, pFrame->height, AV_PIX_FMT_BGR24,
      SWS_BICUBIC, NULL, NULL, NULL);

  // scale the image slice in pFrame->data and put the resulting scaled slice in the image in dst.
  // img_convert_ctx is the scaling context perviously created with sws_getContext().
  // pFrame->data is the array containing the pointers to the planes of the source slice.
  // pFrame->linesize is the array containing the strides for each plane of the source image
  // 0 is the position in the source image of the slice to process, that is the number in the image of the first row of the slice.
  // pFrame->height is the height of the source slice, that is the number of rows in the slice.
  // pFrameBGR->data is the array containing the pointers to the planes of the destination image.
  // pFrameBGR->linesize is the array containing the strides for each plane of the destination image.
  sws_scale(img_convert_ctx,
      pFrame->data,
      pFrame->linesize, 0, pFrame->height,
      pFrameBGR->data,
      pFrameBGR->linesize);
  sws_freeContext(img_convert_ctx);

  int linesize = pFrame->width * 3;
  int height = pFrame->height;

  int stride_0 = height * linesize;
  int stride_1 = linesize;
  int stride_2 = 3;

  // source frame
  // destination frame. arr is an allocated Zero-Value pyArray.
  uint8_t *src = (uint8_t*) pFrameBGR->data[0];
  uint8_t *dest = (uint8_t*) (*arr)->data;

  // The following code copies data from source (pFrameBGR) to dest (a pyArray, which can be accessed in python).
  int array_idx;
  if (cur_pos == pos_target) {
    array_idx = 1;
  } else {
    array_idx = 0;
  }
  memcpy(dest + array_idx * stride_0, src, height * linesize * sizeof(uint8_t));
  av_free(buffer);
}

// This function creates and loads both motion vector and residual maps.
// sd: a pointer to the side data.
// bgr_arr, mv_arr, res_arr are three pyArrays for bgr, mv, and res respectively.
// cur_pos: current position
// accumulate: whether accumulating the residual map and motion vectors.
// representation: either MV or RESIDUAL
// accu_src[x,y,0] indicates the postion of current pixel in the last key-frame on x-axis.
// accu_src[x,y,1] indicates the postion of current pixel in the last key-frame on y-axis.
// accu_src_old has similar utility as accu_src. accu_src_old is used as temp in accumulation.
// width: pFrame->width
// height: pFrame->height
// pos_target: targetting position.
void create_and_load_mv_residual(
  AVFrameSideData *sd,
  PyArrayObject * bgr_arr,
  PyArrayObject * mv_arr,
  PyArrayObject *res_arr,
  int cur_pos,
  int accumulate,
  int representation,
  int *accu_src,
  int *accu_src_old,
  int width,
  int height,
  int pos_target) {

  // Not clear.
  // val_x and val_y are the motion along x axis and y axis respectively.
  int p_dst_x, p_dst_y, p_src_x, p_src_y, val_x, val_y;
  // A data structure containing motion vectors, including the following data fields:
  // int32_t source: where the current macroblock comes from; negative value when it comes from the past, positive value when it comes from the future.
  // uint8_t w: width of the block
  // uint8_t h: height of the block.
  // int16_t src_x: absolute source position
  // int16_t src_y: absolute source position
  // int16_t dst_x: absolute destination position
  // int16_t dst_y: absolute destination position
  // flags: Extra flag information
  // Note: If I am understanding correctly, this structure should be of fixed size, since the data field is fixed.
  const AVMotionVector *mvs = (const AVMotionVector *) sd->data;

  for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
    const AVMotionVector *mv = &mvs[i];
    // This negative value indicates that the motion vector comes from the past.
    // CoViAR has this constraints since it only handles I- and P-frame, while does not support B-frame.
    assert(mv->source == -1);

    // If motion on at least one of the direction (x or y) is non-zero, then do something. Otherwise, skip.
    if (mv->dst_x - mv->src_x != 0 || mv->dst_y - mv->src_y != 0) {

      // val_x is the motion along x axis
      val_x = mv->dst_x -  mv->src_x;
      // val_y is the motion along y axis
      val_y = mv->dst_y - mv->src_y;

      // Seems to loop over all pixels in a block. Not clear why start from a negative value.
      for (int x_start = (-1 * mv->w / 2); x_start < mv->w / 2; ++x_start) {
        for (int y_start = (-1 * mv->h / 2); y_start < mv->h / 2; ++y_start) {
          // Not clear.
          p_dst_x = mv->dst_x + x_start;
          p_dst_y = mv->dst_y + y_start;

          p_src_x = mv->src_x + x_start;
          p_src_y = mv->src_y + y_start;

          if (p_dst_y >= 0 && p_dst_y < height &&
              p_dst_x >= 0 && p_dst_x < width &&
              p_src_y >= 0 && p_src_y < height &&
              p_src_x >= 0 && p_src_x < width) {

              // Write MV.
              if (accumulate) {
                for (int c = 0; c < 2; ++c) {
                  // This line is very interesting. May give some insight on the utility of accu_src and accu_src_old.
                  accu_src        [p_dst_x * height * 2 + p_dst_y*2 + c]
                    = accu_src_old[p_src_x * height * 2 + p_src_y * 2 + c];
                }
              } else {
                *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = val_x;
                *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = val_y;
              }
          }
        }
      }
    }
  }

  if (accumulate) {
    // This line copies width * height * 2 * sizeof(int) bytes from memory area accu_src to memory area accu_src_old.
    memcpy(accu_src_old, accu_src, width * height * 2 * sizeof(int));
  }

  if (cur_pos > 0) {
    // Generate motion vector in the case of accumulate.
    // accu_src[x,y,0] indicates the postion of current pixel in the last key-frame on x-axis.
    // accu_src[x,y,1] indicates the postion of current pixel in the last key-frame on y-axis.
    // Thus x - accu_src[x,y,0] indicates how far the current pixel (x,y) has moved along x-axis.
    // Similarly, y - accu_src[x,y,0] indicates how far the current pixel (x,y) has moved along y-axis.
    if (accumulate) {
      if (representation == MV && cur_pos == pos_target) {
        for (int x = 0; x < width; ++x) {
          for (int y = 0; y < height; ++y) {
            *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0))
            = x - accu_src[x * height * 2 + y * 2];
            *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1))
            = y - accu_src[x * height * 2 + y * 2 + 1];
          }
        }
      }
    }
    if (representation == RESIDUAL && cur_pos == pos_target) {
      // bgr_arr, mv_arr, res_arr are three pyArrays for bgr, mv, and res respectively.
      uint8_t *bgr_data = (uint8_t*) bgr_arr->data;
      int32_t *res_data = (int32_t*) res_arr->data;

      // Define the stride along axis 0, 1, and 2.
      int stride_0 = height * width * 3;
      int stride_1 = width * 3;
      int stride_2 = 3;

      // Not clear the utility.
      int y;

      for (y = 0; y < height; ++y) {
        // Not clear.
        int c, x, src_x, src_y, location, location2, location_src;
        int32_t tmp;
        for　(x = 0; x < width; ++x) {
          // This defines the location of current pixel.
          tmp = x * height * 2 + y * 2;
          if (accumulate) {
            // Intersting. Can be mapped with the accumulation approach in CoViAR. Not clear now.
            src_x = accu_src[tmp];
            src_y = accu_src[tmp + 1];
          } else {
            // (*((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0))) indicates the
            // Not clear.
            src_x = x - (*((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0)));
            src_y = y - (*((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1)));
          }
          // Not clear.
          location_src = src_y * stride_1 + src_x * stride_2;

          // Not clear.
          location = y * stride_1 + x * stride_2;
          for (c = 0; c < 3; ++c) {
            // Not clear.
            location2 = stride_0 + location;
            // Not clear.
            res_data[location] = (int32_t) bgr_data[location2]
                               - (int32_t) bgr_data[location_src + c];
            // Not clear.
            location += 1;
          }
        }
      }
    }
  }
}





int decode_video(
  int gop_target,
  int pos_target,
  PyArrayObject ** bgr_arr,
  PyArrayObject ** mv_arr,
  PyArrayObject ** res_arr,
  int representation,
  int accumulate) {

  AVCodec *pCodec;
  AVCodecContext *pCodecCtx=NULL;
  AVCodecParserContext *pCodecParserCtx=NULL;

  // File handler
  FILE *fp_in;
  // This structure describes decoded (raw) audio or video data. Note that this only allocates the AVFrame itself, the buffers for the data must be managed through other means (see below).
  AVFrame *pFrame;
  // This structure describes decoded (raw) audio or video data.
  AVFrame *pFrameBGR;

  // This buffer is used to store raw data read from the video stream.
  const int in_buffer_size=4096;
  uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];
  // The memset() function fills the first FF_INPUT_BUFFER_PADDING_SIZE of the memory area
  // pointed to by (in_buffer + in_buffer_size ) with the constant byte 0.
  memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

  uint8_t *cur_ptr;
  int cur_size;
  // This structure stores compressed data. For video, it should typically contain one compressed frame.
  AVPacket packet;
  // Indicating status.
  int ret, got_picture;

  avcodec_register_all();

  // Find a registered decoder with a matching codec ID.
  // :param AV_CODEC_ID_MPEG4: AV_CODEC_ID_MPEG4 is the AVCodecID of the requested decoder.
  // :return: A decoder if one was found, NULL otherwise.
  pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);
  if (!pCodec) {
    printf("Could not allocate video codec context\n");
    return -1;
  }

  // It seems that codec, codec context, and codec parser
  //    can be used together to decode a video.
  pCodecCtx = avcodec_alloc_context3(pCodec);
  if (!pCodecCtx){
      printf("Could not allocate video codec context\n");
      return -1;
  }

  pCodecParserCtx=av_parser_init(AV_CODEC_ID_MPEG4);
  // pCodecParserCtx=av_parser_init(AV_CODEC_ID_H264);
  if (!pCodecParserCtx){
      printf("Could not allocate video parser context\n");
      return -1;
  }

  // Simple key:value store.
  AVDictionary *opts = NULL;
  // Set the given entry in opts, overwriting an existing entry. Here, opts is a pointer to a AVDictionary,
  // "flags2" is a key
  // "+export_mvs" is used as the previous matching element to find the next. If set to NULL, the first matching element is returned.
  // 0 represents an AV_DICT_* flags controlling how the entry is retrieved.
  av_dict_set(&opts, "flags2", "+export_mvs", 0);
  // Initilize the AVCodecContext to use the given AVCodec.
  if (avcodec_open2(pCodecCtx, pCodec, &opts) < 0) {
    printf("Could not open codec\n");
    return -1;
  }

  // Open a handler for the video file. Note that filename is a global variable holding the video file name.
  fp_in = fopen(filename, "rb");
  if (!fp_in) {
      printf("Could not open input stream\n");
      return -1;
  }

  int cur_pos = 0;

  // Allocate an AVFrame and set its fields to default values
  pFrame = av_frame_alloc();
  pFrameBGR = av_frame_alloc();

  // a buffer storing image data.
  uint8_t *buffer;

  // This function initilizes optional fields of a packet with default values.
  // packet is an AVPacket.
  av_init_packet(&packet);

  int *accu_src = NULL;
  int *accu_src_old = NULL;

  while (1) {

    // This function reads in_buffer_size items of data, each 1 bytes long,
    // from the stream pointed to by fp_in, where fp_in is a file descriptor
    // and storing them at the location given by in_buffer.
    cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
    if (cur_size == 0)
      break;
    cur_ptr=in_buffer;

    while (cur_size>0) {

      // This function parses a packet.
      // pCodecParserCtx is the parser context.
      // pCodecCtx is the codec context.
      // packet.data stores parsed buffer or NULL if not yet finished.
      // packet.size is set to size of parsed buffer or zero if not yet finished.
      // cur_ptr is the input buffer
      // cur_size is the input length, to signal EOF, this should be 0 (so that the last frame can be output).
      // The last three values represent input presentation timestamp, input decoding timestamp, and input byte position in sream. Note that AV_NOPTS_VALUE represents Undefined timestamp value.
      // return the number of bytes of the input bitstream used.
      int len = av_parser_parse2(
          pCodecParserCtx, pCodecCtx,
          &packet.data, &packet.size,
          cur_ptr, cur_size,
          AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);

      cur_ptr += len;
      cur_size -= len;

      if(packet.size==0)
        continue;

      // Type of frame. Could be one of I-frame (AV_PICTURE_TYPE_I), P-frame (AV_PICTURE_TYPE_P), B-frame (AV_PICTURE_TYPE_B), etc.
      // ++cur_gop since I-frame indicates that a new gop starts.
      if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {
        ++cur_gop;
      }

      // We have find the desired gop. Now, start to identify the desired position.
      // This position is between 0~11, indicating the position inside current gop.
      if (cur_gop == gop_target && cur_pos <= pos_target) {

        // This line decodes the video frame of size packet->size from packet->data into pFrame.
        // Here, pFrame is a pointer to an AVFrame code.
        // got_picture is an integer. Zero if no frame could be decompressed, otherwise, it is nonzero.
        // packet is the input AVPacket containing the input buffer
        // Return: On error a negative value is returned, otherwise the number of bytes used or zero if no frame could be decomrpessed.
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
        if (ret < 0) {
          printf("Decode Error.\n");
          return -1;
        }
        int h = pFrame->height;
        int w = pFrame->width;

        // Initialize arrays,
        if (! (*bgr_arr)) {
          npy_intp dims[4];
          dims[0] = 2;
          dims[1] = h;
          dims[2] = w;
          dims[3] = 3;
          *bgr_arr = PyArray_ZEROS(4, dims, NPY_UINT8, 0);
        }

        if (representation == MV && ! (*mv_arr)) {
          npy_intp dims[3];
          dims[0] = h;
          dims[1] = w;
          dims[2] = 2;
          *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
        }

        if (representation == RESIDUAL && ! (*res_arr)) {
          npy_intp dims[3];
          dims[0] = h;
          dims[1] = w;
          dims[2] = 3;

          *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
          *res_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
        }


        // This code is used when accumulation is required.
        // If accumulate, then initialize accu_src and accu_src_old.
        if ((representation == MV ||
             representation == RESIDUAL) && accumulate &&
            !accu_src && !accu_src_old) {

            accu_src    = (int*) malloc(w * h * 2 * sizeof(int));
            accu_src_old = (int*) malloc(w * h * 2 * sizeof(int));

            for (size_t x = 0; x < w; ++x) {
              for (size_t y = 0; y < h; ++y) {
                accu_src_old[x * h * 2 + y * 2    ] = x;
                accu_src_old[x * h * 2 + y * 2 + 1] = y;
              }
              // This line copies h*w*2*sizeof(int) bytes from memory area accu_src to memory area accu_src_old.
              memcpy(accu_src, accu_src_old, h * w * 2 * sizeof(int));
            }

            // A non-zero got_picture means that a pFrame contains a decompressed frame.
            if (got_picture) {

              // Get the base frame at three contitions.
              if ((cur_pos == 0              && accumulate && representation == RESIDUAL) ||
                  (cur_pos == pos_target - 1 && !accumulate & representation == RESIDUAL) ||
                  cur_pos == pos_target) {
                  // pFrame is an AVFrame containing a decompressed frame.
                  // pFrameBGR is another AVFrame, but has not been used yet.
                  // buffer: A pointer to image buffer.
                  // bgr_arr: allocated Zero-Value pyArray.
                  // cur_pos: Current position.
                  // pos_target: Desired position.
                  create_and_load_bgr(
                      pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, pos_target);
              }


              // Enter here if we want to retrieve motion vectors and residual maps.
              if (representation == MV ||
                  representation == RESIDUAL) {
                  // A pointer to the side data.
                  AVFrameSideData *sd;
                  // Return a pointer to the side data of a given type on success.
                  // NULL if there is no motion vector in this frame.
                  sd = av_frame_get_side_data(pFrame, AV_FRAME_DATA_MOTION_VECTORS);
                  if (sd) {
                      if (accumulate || cur_pos == pos_target) {
                          // This function creates and loads both motion vector and residual maps.
                          // sd: a pointer to the side data.
                          // bgr_arr, mv_arr, res_arr are three pyArrays for bgr, mv, and res respectively.
                          // cur_pos: current position
                          // accumulate: whether accumulating the residual map and motion vectors.
                          // representation: either MV or RESIDUAL
                          // w: pFrame->width
                          // h: pFrame->height
                          // pos_target: targetting position.
                          create_and_load_mv_residual(
                            sd,
                            *bgr_arr, *mv_arr, *res_arr,
                            cur_pos,
                            accumulate,
                            representation,
                            accu_src,
                            accu_src_old,
                            w,
                            h,
                            pos_target);
                      }
                  }
              }
              cur_pos ++;
            }
      }
    }
  }

  //TO BE CONTINUED


  // Flush Decoder
  //End of stream situations. These require "flushing" (aka draining) the codec, as the codec might buffer multiple frames or packets internally for performance or out of necessity (consider B-frames). This is handled as follows:
  // Note the desired gop or pos may be at the very end of the video frame. In this case, nothing can be read by file handler thus the previous huge while loop will break. Meanwhile, the desired frame is stored in the codec buffer.
  // The following code snippet handles this edge case.
  packet.data = NULL
  packet.size = 0;
  while (1) {
    // This line decodes the video frame of size packet->size from packet->data into pFrame.
    ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);

    if (ret < 0) {
      printf("Decode Error. \n")
      return -1;
    }
    if (!got_picture) {
      break;
    } else if (cur_gop == gop_target) {
      if ((cur_pos == 0 && accumulate) ||
          (cur_pos == pos_target - 1 && !accumulate) ||
          cur_pos == pos_target) {
          create_and_load_bgr(
              pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, pos_target);
      }
    }
  }

  // Clean up. Nothing interesting.

  fclose(fp_in);

  av_parser_close(pCodecParserCtx);

  av_frame_free(&pFrame);
  av_frame_free(&pFrameBGR);
  avcodec_close(pCodecCtx);
  av_free(pCodecCtx);
  if ((representation == MV ||
       representation == RESIDUAL) && accumulate) {
      if (accu_src) {
        free(accu_src);
      }
      if (accu_src_old) {
        free(accu_src_old);
      }
  }

  return 0;
}

