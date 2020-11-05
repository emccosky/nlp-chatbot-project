import typing

LOG_LVLS = ["VERBOSE", "DEBUG", "ORPHEUS"]

class User:

    name: str
    spotify_username: str
    genre_prefs: typing.List[bool]
    artist_likes: typing.List[str]
    song_likes: typing.List[str]

    # Ideally these would be part of a track/artist object but for now are separate
    # because they are only used for recommendation seeding
    artist_ids: typing.List[str]
    song_ids: typing.List[str]

    def __init__(self, name, spotify_username, genre_prefs, artist_likes, artist_ids, song_likes, song_ids):
        self.name = name
        self.spotify_username = spotify_username
        self.genre_prefs = genre_prefs
        self.artist_likes = artist_likes
        self.artist_ids = artist_ids
        self.song_likes = song_likes
        self.song_ids = song_ids

    def display(self):
        print("\nName:", self.name)
        print("\nspotify_username:", self.spotify_username)
        print("\ngenre_prefs:", self.genre_prefs)
        print("\nartist_likes:", self.artist_likes)
        print("\nartist_dislikes:", self.artist_dislikes)
        print("\nsong_likes:", self.song_likes)
        print("\nsong_dislikes:", self.song_dislikes)


def log(msg, level, env_mode):
    if (LOG_LVLS.index(level) >= LOG_LVLS.index(env_mode)):
        print("[{lvl}] {out}".format(lvl = level, out = msg))