
def get_video_meta(filename):
    # reads video file and returns width, height and fps 
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = int(video_info['avg_frame_rate'].split('/')[0])
    try:
        nb_frames = int(video_info['nb_frames'])
    except:
        nb_frames = None
    return width, height, fps, nb_frames

def get_video_meta_all(filename):
    # reads video file and returns width, height and fps 
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return  video_info

def read_single_frame(in_filename, frame_num, width, height):
    # reads single frame from an input video and returns the frame as numpy array
    # src - input video
    # frame_num - frame number to return
    out, err = (
    ffmpeg
    .input(in_filename)
    .filter('select','gte(n,{})'.format(frame_num))
    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', preset='ultrafast')
    .run(capture_stdout=True)
    )
    return  np.frombuffer(out, 'uint8').reshape([-1, height, width, 3])[0]

def start_ffmpeg_process1( in_filename):
    # process1 reads video frames sequancially and temporary saves them in buffer
#     logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', crf=23, preset='ultrafast', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_process2( out_filename, width, height):
    # process2 obtains frames from buffer and writies them sequencially
    # produces final output video       
#     logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, crf=23, preset='ultrafast', movflags='faststart', pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(process1, width, height):
    # starts reading frames using subprocessor from buffer
#     logger.debug('Reading frame')

    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def write_frame(process2, frame):
    # writes processed frames into another buffer simultanusly 
    # while reading frames, processed frames are simultanusly written into buffer to produce output file faster
#     logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )




def run_video_processor(in_filename, out_filename, batch_size, show_big_cation, show_face_caption, show_boxes, no_process_real_frame):
    
    logger = get_logger('run_video_processor')
    old_time = time.time()
    width, height, fps, nb_frames = get_video_meta(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)

    # REQUEST_TACKER[ request_id ] = {process1.pid, process1.pid}

    tmp_frames, tmp_faces, tmp_face_cordinates, tmp_num_faces, all_predictions  = [],[],[],[],[]
    is_face_in_video, is_fake_face = False, False

    while True:
        in_frame = read_frame(process1, width, height)

        if in_frame is None:
            logger.info('End of input stream')
            break

        tmp_frames.append(in_frame)
        frame_as_pil = Image.fromarray(in_frame.astype(np.uint8))
        aligned_faces, face_predictions = face_utils.is_face_exsits(frame_as_pil)
        tmp_face_cordinates.append(face_predictions[0])
        tmp_num_faces.append(len(aligned_faces) if aligned_faces is not None else None)

        if aligned_faces is not None:
            for face in aligned_faces:
                f = face.permute(1,2,0).int().numpy().astype(np.uint8)
                tmp_faces.append(f)

        if len(tmp_faces) > batch_size:
            res_tensor = is_fake(tmp_faces)
            fakeness_predictions = decode_predictions(res_tensor)
            
            for fp1, fp2 in fakeness_predictions:
                all_predictions.append(fp1)
                all_predictions.append(fp2)
                
            tmp_face_frame_indexes = [i for i, val in enumerate(tmp_face_cordinates) if val is not None]
            tmp_frames_after_process = []
            prev = 0
            for i in range(len(tmp_frames)):
                if i in tmp_face_frame_indexes:
                    last = prev + tmp_num_faces[i]
                    fakeness_results = fakeness_predictions[prev:last]
                    prev = last
                    processed_frames = process_frames(tmp_frames[i], tmp_face_cordinates[i], fakeness_results, None, show_big_cation,  show_face_caption, show_boxes, no_process_real_frame)
                    tmp_frames_after_process.append(processed_frames)
                else:
                    tmp_frames_after_process.append(tmp_frames[i])

            for frame in tmp_frames_after_process:
                write_frame(process2, frame)
            tmp_frames, tmp_faces, tmp_face_cordinates, tmp_num_faces= [],[],[],[]    
            is_face_in_video = True

    if is_face_in_video is False:
        tmp_frames, tmp_faces, tmp_face_cordinates, tmp_num_faces = [],[],[],[]
        logger.info('No face detected in the video. Process has stopped')
        process1.kill()
        process2.kill()
        time_spent = time_parser(time.time() - old_time)
        return  ( in_filename, in_filename, '0', 'No face detected', time_spent)

    logger.info('Waiting for input stream')
    process1.wait()
    logger.info('Waiting for output stream')
    process2.stdin.close()
    process2.wait()

    fakes = all_predictions.count('fake')
    reals = len(all_predictions)-fakes
    result = 'No Deepfake Detected' if fakes == 0 else 'Deepfake Detected'
    percent = str((100 * fakes) / len(all_predictions))[:4] if len(all_predictions) != 0 else ''
    
    new_time =  time.time()
    time_spent = time_parser(new_time - old_time)
    logger.info('Done')
    logger.info('Processing time: {}'.format(time_spent))
    return  (in_filename, out_filename, percent, result, time_spent)