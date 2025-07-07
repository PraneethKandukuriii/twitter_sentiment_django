from django import forms

class TweetForm(forms.Form):
    tweet = forms.CharField(label='Enter a tweet', max_length=280)
