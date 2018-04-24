## Folder for json_dict

This is the folder for storing json dictionaries of training, validating, testing datasets. Here is what a line of the json dict should look like:

{"duration": 9.915, "text": "shortly after passing one of these chapels we came suddenly upon a village which started up out of the mist and i was alarmed lest i should be made an object of curiosity or dislike", "key": ".././data/LibriSpeech/dev-clean/2412/153954/2412-153954-0000.wav"}


Duration shows the duration of the sentence. Text is the ground truth for transcription. And "key" indicates the path of the audio file corresponding to the text transcription. 
