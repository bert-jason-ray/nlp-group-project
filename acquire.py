"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['jagrosh/MusicBot',
    'Just-Some-Bots/MusicBot',
    'SudhanPlayz/Discord-MusicBot',
    'IVETRI/SongPlayRoBot',
    'Splamy/TS3AudioBot',
    'galnir/Master-Bot',
    'szastupov/musicbot',
    'umutxyp/MusicBot',
    'AsmSafone/MusicPlayer',
    'DarkoPendragon/discord.js-musicbot-addon',
    'sosedoff/musicbot',
    'Allvaa/lavalink-musicbot',
    'AnonymousR1025/FallenMusic',
    'NotAWeebDev/Misaki',
    'xCrypt0r/Crucian',
    'NithishCodez/Discordjs-MusicBot',
    'Hazmi35/jukebox',
    'hydrox19/discord-music',
    'Garlic-Team/MusicBot',
    'TheVaders/MusicBot',
    'SinusBot/docker',
    'vigneshd332/proximity',
    'parasop/NEW-MUSIC-BOT',
    'kijk2869/discodo',
    'ljgago/MusicBot',
    'LordReaperY/MusicBot',
    'ZeroDiscord/MusicBot',
    'TamilBots/TamiliniMusic',
    'Pranav6966/v13-musicbot-withbuttons',
    'kabirsingh2004/lavalink-music-bot-2021',
    'kira0x1/mikaela',
    'SiruBOT/SiruBOT',
    'mrmotchy/highend-musicbot',
    'SuperSandro2000/docker-images',
    'LOGI-LAB/music-video-streamer',
    'kaaaxcreators/Discord-MusicBot',
    'minichris/MusicBot',
    'Bettehem/ts3-musicbot',
    'YaBoyWonder/MusicBot',
    'BluSpring/discord.js-lavalink-musicbot',
    'AdrienPensart/musicbot',
    'Ayush4385/Aoi.js-bot',
    'kijk2869/discodo.js',
    'Spidey-Org/Spidey',
    'Adivise/DisSpaceX',
    'craumix/jmb-container',
    'CarterGunale/MusicBot',
    'p-fruck/jim',
    'PapyrusThePlant/Panda',
    'xCuzSkillz/MusicBot-Premium-Website',
    'StrapBot/StrapBot',
    'MusicBottle/MusicBottle',
    'KingRain/SimpleDiscordMusicBot',
    'freedmand/tapcompose',
    'brendonmiranda/clancy',
    'Rdimo/Simple-Discord-Music-Bot',
    'baizel/MusicBot-With-Spotify-For-Discord',
    'tooxo/Geiler-Musik-Bot',
    'Gabriel-M-Martins/Discord-MusicBot',
    'MR-INVISIBLEBOY/LEGENDBOT-INVISIBLE1',
    'biswajitguha01/NSFW-Bot',
    'karyeet/Mandarine',
    'parasop/discord.js-musicbot',
    'LucunJi/kaiheila-musicbot',
    'BearOffice/MusicBot',
    'svenwiltink/go-MusicBot',
    'dmuth/musicbot-docker',
    'stevekinney/musicbot',
    'Sync-Codes/Ajax',
    'r3boot/go-musicbot',
    'Budovi/ddmbot',
    'Music-Bot-for-Jitsi/Jimmi',
    'itsayushch/musico',
    'PollieDev/MusicMaistro',
    'parzuko/elvis',
    'hemreari/feanor-dcbot',
    'spencer911/Discord-Music-Bot-Advanced-Rhtm-Mee6',
    'KunalBagaria/musicbot',
    'BjoernPetersen/MusicBot',
    'stuyy/MusicBot',
    'PawanBro/MusicBot',
    'PredatorHackerzZ/VC-Streamer',
    'cFerg/MusicBot',
    'Manebot/ts3',
    'JoaoOtavioS/discord.js-musicbot',
    'Heavyrisem/Discord_MusicBot',
    'r-Norge/ShiteMusicBot',
    'etcroot/Mimi',
    'Romilchavda/Foxy-Music',
    'johann-lr/cadence-v2',
    'TysonOP/Erela.js-Music',
    'xtreameprogram/MusicBotTut',
    'james58899/MusicBot',
    'TheOnlyArtz/MusicBot',
    'nikosszzz/musicbot',
    'JovemHero/MusicBot',
    'TimovNiedek/MusicBot',
    'yousukeayada/discord-musicbot-citron',
    'Manevolent/ts3j-musicbot',
    'rexlManu/ts3audiobot',
    'BjoernPetersen/Kiu',
    'Nich87/Discord-Musicbot',
    'Malgnor/musicbotdosbrodi',
    'pheonic/MusicBot',
    'saltukozelgul/MusicBot',
    'abd-ar/discord-musicbot',
    'playteddypicker/discordMusicbotTemplate',
    'iamsurojit/Discord-MusicBot',
    'Doeca/musicBotForCanteen',
    'Weebs-Kingdom/Yuki-Sora',
    's-vivien/BlindTestBot',
    'ScalaStudios/ScalaPublic',
    'francescomosca/innova-discord-bot',
    'DemonKingSwarn/DemonCord-Music',
    'andasilva/MusicBot',
    'HTSTEM/MusicBot',
    'rlp81/musicbot',
    'Yaamiin/KennedyXMusic',
    'wvffle/ts3-musicbot',
    'Manebot/manebot',
    'Yaamiin/forhim',
    'astronautlevel2/Amadeus',
    'MOZGIII/musicbot2',
    'NABLYOUN/-',
    'gauthamp10/musicbot',
    'brainboxdotcc/musicbot',
    'MOZGIII/musicbot',
    'Yaamiin/musicbottest',
    'abdulsamedkeskin/MusicBot',
    'GceCold/MusicBot',
    'Hero-Inc/MusicBot',
    'NishantMajumdar2/musicBot',
    'rinsuki-lab/musicbot-ts',
    'fan87/wd-musicbot',
    'javaj0hn/TeamSpeak-MusicBot',
    'atomicnetworkseu/atomicradio-discordbot',
    'nosesisaid-archive/music-24-7',
    'kaoru-nk/atmusicbot',
    'NinjaBros/Music-Bot',
    'botdevelopersbrofc/botmusictraduzido',
    'N3bby/Shimarin',
    'MadBoy-X/VC-Bot',
    'RHGDEV/js-musicbot',
    'ZauteKm/MusicBot',
    'R3dlessX/ARCHIVED-Discord-MusicBot',
    'jonydaimary/musicbot',
    'Xirado/Tuner',
    'Kaufisch/Discord_24-7_MusicBot',
    'LingleDev/Hulk_Moosic',
    'Ryuukai/discord-musicbot',
    'respoke/chan_respoke-musicbot-example',
    'decentboyy/Octave_MusicBot',
    'Sueqkjs/Nelly',
    'cocoastorm/cocoabot',
    'crazyczy/musicbot',
    'ViperXD/MUSICBOT',
    'NEZH69/musicbot',
    'alantomjose/Musicbot',
    'jeffreymkabot/musicbot',
    'amigos2007/musicbot',
    'rexlManu/BetterAudioBot',
    'inox9/musicBot',
    'Snowclub111/MusicBot',
    'evgeniy-btw/MusicBot',
    'macman31/MusicBot',
    'dmuth/MusicBot-Ansible',
    'kijk2869/MusicBot',
    'jadedtdt/MusicBot',
    'hirusha-adi/MusicBot',
    'PineappleTurnovers/MusicBot',
    'FelipeKreulich/MusicBot',
    'iFanpSGTS/MusicBot',
    'rocats/musicBot',
    'Aahanbhatt/telegram-musicbot',
    'pranavgoyanka/Reddit-Musicbot',
    'yemix/user-musicbot',
    'tjhorner/mumble-musicbot',
    'Cat-of-Tg/MusicBotTest',
    'TR-TECH-GUIDE/Discord-MusicBot',
    'FlyingndCoding/-MusicBot-discord',
    'playerdecuple/Custom-MusicBot-Maker',
    'leary1337/MusicBotVk',
    'thundric1/Thundric-x-MusicBot',
    'huifeiderouge/MusicBot-by-Producer-404',
    'typekcz/beepboop-steam',
    '12yuens2/r.a.dio-discord-bot',
    'FabiChan99/24-7-discord-radio',
    'poco0317/BarinadeBot-Rewrite',
    'Xetera/ThiccBot.py',
    'Shadowsith/mumble-ruby-pluginbot-plugins',
    'Jimgersnap/DJ-Roomba',
    'ItsClairton/Anny',
    'Micium-Development/Bounce',
    'philliphqs/hqs.bot',
    'shoemakk/MusicBot-Pub',
    'Davidremo02/LazyMusicbot',
    'MrRizoel/RiZoeLXMusic',
    'noirscape/MusicBot-2',
    'bhkvlldu/MusicBot',
    'niccholaspage/MusicBot']

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)