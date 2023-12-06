from decimal import ROUND_HALF_UP, Decimal
from typing import Union

import soundfile as sf
import torchaudio

from utils import handle_scp

Seconds = float
Decibels = float


def compute_num_samples(
    duration: Seconds, sampling_rate: Union[int, float], rounding=ROUND_HALF_UP
) -> int:
    """
    Convert a time quantity to the number of samples given a specific sampling rate.
    Performs consistent rounding up or down for ``duration`` that is not a multiply of
    the sampling interval (unlike Python's built-in ``round()`` that implements banker's rounding).
    """
    return int(
        Decimal(round(duration * sampling_rate, ndigits=8)).quantize(
            0, rounding=rounding
        )
    )


def read_wav(fname, return_rate=False, sr=8000, start=None, end=None):
    """
    Read wavfile using Pytorch audio
    input:
          fname: wav file path
          return_rate: Whether to return the sampling rate
    output:
           src: output tensor of size C x L
                L is the number of audio frames
                C is the number of channels.
           sr: sample rate
    """
    if start != None:
        frame_offset = compute_num_samples(start, sr)
    else:
        frame_offset = 0
    if end != None:
        num_frames = compute_num_samples(end - start, sr)
    else:
        num_frames = -1
    src, sr = torchaudio.load(
        fname,
        num_frames=num_frames,
        frame_offset=frame_offset,
        channels_first=True,
    )
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


def write_wav(fname, src, sample_rate):
    """
    Write wav file
    input:
          fname: wav file path
          src: frames of audio
          sample_rate: An integer which is the sample rate of the audio
    output:
          None
    """
    sf.write(fname + ".wav", src, sample_rate)


class AudioReader(object):
    """
    Class that reads Wav format files
    Input as a different scp file address
    Output a matrix of wav files in all scp files.
    """

    def __init__(self, scp_path, sample_rate=8000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())

    def load(self, key, start=None, end=None):
        src, sr = read_wav(
            self.index_dict[key],
            return_rate=True,
            sr=self.sample_rate,
            start=start,
            end=end,
        )
        if self.sample_rate is not None and sr != self.sample_rate:
            raise RuntimeError(
                "SampleRate mismatch: {:d} vs {:d}".format(sr, self.sample_rate)
            )
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
            yield key, self.load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(index, num_uttrs)
                )
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self.load(index)


if __name__ == "__main__":
    r = AudioReader("/home/likai/data1/create_scp/cv_s2.scp")
    index = 0
    print(r[1])
