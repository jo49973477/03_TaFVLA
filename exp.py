import sys
import av

with av.open(sys.argv[1]) as container:
    frame = next(container.decode(video=0))
    image = frame.to_ndarray(format="rgb24")

    print("Decoded:", image.shape)
    print("PTS:", frame.pts)
    print("Time:", frame.time)