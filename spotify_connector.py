import chatbot_lib as lib
from chatbot_lib import User

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import json

SPOTIPY_CLIENT_ID = '28e288eb88f342c79ac8389d6ab32529'
with open('client.secret') as f:
    SPOTIPY_CLIENT_SECRET = f.read()
SPOTIPY_REDIRECT_URI = "http://localhost:9090/"


def refresh_token(username):

    scope = 'user-library-read user-top-read'

    if(SPOTIPY_CLIENT_SECRET):
        token = util.prompt_for_user_token( username, 
                                            scope, 
                                            client_id = SPOTIPY_CLIENT_ID, 
                                            client_secret = SPOTIPY_CLIENT_SECRET, 
                                            redirect_uri = SPOTIPY_REDIRECT_URI)
    else:
        print("ERROR CONNECTING TO SPOTIFY. Check client.secret file")
        quit()
    
    return token


def hydrate_user(user: User):
    token = refresh_token(user.spotify_username)
    tracks = []
    track_ids = []
    artists = []
    artist_ids = []
    if token:
        sp = spotipy.Spotify(auth=token)
        
        results = sp.current_user_top_tracks()
        for item in results['items']:
            tracks.append("Title: {name} Artist: {artist}".format(name = item['name'], artist = item['artists'][0]['name']))
            track_ids.append(item['id'])
        
        results = sp.current_user_top_artists(time_range='long_term')
        for item in results['items']:
            artists.append("{artist}".format(artist = item['name']))
            artist_ids.append(item['id'])
    else:
        print("Can't get token for", user.spotify_username)
    user.song_likes = tracks
    user.artist_likes = artists
    user.song_ids = track_ids
    user.artist_ids = artist_ids
    return user


def get_recommendations(user: User):
    token = refresh_token(user.spotify_username)
    recs = []
    if token:
            sp = spotipy.Spotify(auth=token)
            results = sp.recommendations(seed_artists = [user.artist_ids[0]], seed_tracks = [user.song_ids[0]])
            #print(results)
            for track in results['tracks']:
                recs.append("Title: {name} Artist: {artist}".format(name = track['name'], artist = track['artists'][0]['name']))
    else:
        print("Can't get token for", user.spotify_username)
    return recs        