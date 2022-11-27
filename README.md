# sound-aid-v0.3
3rd version of the sound-aid app. This is still a prototype.

This is still a prototype.
The app will let you pick a file from the 'input' folder.
It assumes you have a number of '.wav' files in there.

It will make a prediction based on the mean amplitude of the waveform of the audio file.
mean_amp_times_1000 > -0.2 and mean_amp_times_1000 <= 0  will result in a prediction of: SPEECH
Any value outside of that range will lead to a classification of: MUSIC

This algorithm isn't perfect. For really small speech files it will not work.
