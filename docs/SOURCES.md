# Source selection guide

How we choose YouTube videos for the dataset, what copyright allows, which channels are
candidates, and how the vocabulary will be chosen. Decisions here are summarised in
`docs/RESEARCH_LOG.md` (D15–D17).

---

## 1. What makes a video usable

The pipeline keeps a word clip only if one face is visible for the whole 1.16 s, is at least
100 px wide, is turned less than 30° and tilted less than 40°, and no shot cut happens inside the
clip (README, Stage 4). A video is worth downloading when most of its speech meets that:

| Good | Poor |
|---|---|
| news anchors reading to camera | dramas and serials (acting, music, many cuts) |
| one-on-one interviews and podcasts with close-up shots of each speaker | panels shown in wide shots, split screens |
| lectures, speeches, talks (TEDx) | slideshows, screen recordings, voice-over on B-roll |
| vlogs with the speaker facing the camera | music videos, reaction compilations, crowds |
| 720p or higher | dubbed content, heavy background music |
| natural Bengali speech | captions burned in over the mouth |

**Use official channels only.** Search results include many re-upload channels, for example
channels re-posting BBC bulletins or radio news. Their content isn't theirs, so any licence they
show is meaningless, and the same footage would appear twice in the dataset.

Radio news posted as video (e.g. Akashvani bulletins) has no visible speaker and is useless here.

## 2. What other datasets used

| Dataset | Source material |
|---|---|
| LRW | BBC TV programmes; the paper's examples are *Question Time* and *BBC News at One* |
| LRS3 | TED and TEDx talks |
| LRW-1000 | 26 broadcast sources, 51 news and conversational programmes |
| LRW-AR | news programmes on YouTube |
| LRW-Persian | 67 TV programmes (1,989 h) |
| MultiVSR (Prajwal et al. 2025) | ~123k YouTube videos, IDs taken from AVSpeech |

None of them restrict themselves to one genre. What they share is **talking heads with natural
speech**, mostly broadcast. Mixing genres (news, talk, podcast, lecture) gives more speakers,
speaking styles, vocabulary and recording conditions. The face filters keep the visual format
consistent. **Recommendation:** mix genres, cap how much any one channel or speaker contributes,
and report the genre split in the paper.

## 3. Copyright: what we can and cannot do

*This is a practical summary, not legal advice. Confirm the release plan with the university's
legal or ethics office.*

- **Almost everything on YouTube is copyrighted.** Videos use the *Standard YouTube License*
  unless the uploader chooses Creative Commons. "No copyright" videos essentially don't exist
  outside public-domain or CC material.
- **YouTube's Creative Commons option is CC BY:** anyone may copy, share and adapt the video,
  including commercially, with attribution (YouTube Help, "License types on YouTube").
  **CC BY clips could be redistributed** with credit. This only holds if the uploader owns the
  content: a re-upload marked CC grants nothing.
- **TEDx talks are CC BY-NC-ND:** sharing with attribution is allowed, but no commercial use and
  **no derivatives** (TEDx licence pages). Cutting a talk into clips is a derivative, so we can use
  TEDx for research and release IDs, but we can't share our clips.
- **Standard-licence videos:** LRS3, VoxCeleb, AVSpeech and MultiVSR are built from such videos
  for research and released as **YouTube IDs and timestamps** (VoxCeleb and AVSpeech) or under
  research-only terms (LRS3). Whether this is allowed depends on the country, for example fair use
  in the US or text-and-data-mining exceptions elsewhere. We follow the same practice: release
  IDs, frame ranges, crop boxes and labels; share clips only on request for research; offer
  takedown.
- **Automatic check:** yt-dlp's `license` field is filled only when YouTube shows a licence row,
  which it does for Creative Commons. It is empty for standard-licence videos. `info.json` records
  it for every download.

**Proposed tiers:**

| Tier | Licence | What we release |
|---|---|---|
| A | CC BY, uploaded by the owner | clips + metadata (e.g. a fully shareable test or sample set) |
| B | standard YouTube licence, or TEDx | metadata only (IDs, frames, boxes, labels) |

Few Bengali talk channels are CC BY. The search found 3 out of 37 (table below), so Tier B
will be most of the data.

## 4. Candidate channels (metadata checked 2026-10-01)

Found with yt-dlp metadata searches (Bengali and English queries for news, talk shows, podcasts,
interviews, lectures, TEDx, plus YouTube's Creative Commons filter). For 1–2 long videos per
channel, the licence, YouTube's language tag, follower count and resolution were read. **No
video was downloaded or watched.**

Caveats:
- **Not screened visually.** Whether speakers face the camera, how often the camera cuts, and
  host gender are unknown until screening (§6).
- **Region is a guess** from the channel name and titles.
- **YouTube's language tag is unreliable:** Prothom Alo is tagged `hi`, EKHON TV and The Talk Show
  `en`. A language-ID check per video is needed (MultiVSR used a VoxLingua-trained model).
- **TEDx Talks** is a multi-language channel; take only the Bengali talks.
- The two **religious-lecture** channels are CC BY, but the speech is mostly from a few male
  preachers with a narrow vocabulary. Use them sparingly, if at all.
- ✓ = YouTube-verified channel.

| Channel | Type | Region | Licence | Followers | Max res. | YouTube language tag | Channel videos | Sample video IDs |
|---|---|---|---|---|---|---|---|---|
| ATN Bangla News ✓ | news | BD | standard | 10.40 M | 480p | bn | [link](https://www.youtube.com/channel/UCbgcYEdMsuypG2NJ-znBp3w/videos) | G0oaPXBF5BY, ZHkHl600ruw |
| BanglaVision NEWS ✓ | news | BD | standard | 10.00 M | 1080p | bn | [link](https://www.youtube.com/channel/UCA4y9g_GRdNrZLH_u5LLSdw/videos) | URT8-e8mADE, q6jw27C-Clk |
| BBC News বাংলা ✓ | news | BD/intl | standard | 6.57 M | 1080p | None | [link](https://www.youtube.com/channel/UChPD2BQ2dJ-w00klQQkc-Qg/videos) | v0jl5zTz4xU, GvBDC_mNDgM |
| Jamuna TV ✓ | news | BD | standard | 31.00 M | 720p | bn | [link](https://www.youtube.com/channel/UCN6sm8iHiPd0cnoUardDAnw/videos) | CXxmbuE0_ik, X_mYhkdZwR0 |
| EKHON TV ✓ | news | BD | standard | 4.33 M | 1080p | en | [link](https://www.youtube.com/channel/UCWVqdPTigfQ-cSNwG7O9MeA/videos) | 00IUSRe5iKk |
| Independent Television ✓ | news | BD | standard | 12.50 M | 1080p | bn | [link](https://www.youtube.com/channel/UCATUkaOHwO9EP_W87zCiPbA/videos) | espGALJaIf4, LAdpn97X1_Y |
| Ekattor TV ✓ | news | BD | standard | 16.00 M | 1080p | bn | [link](https://www.youtube.com/channel/UCtqvtAVmad5zywaziN6CbfA/videos) | 7HfLvnkrzhU, uwz45SlkfXc |
| DW বাংলা ✓ | news | BD/intl | standard | 930 k | 2160p | None, bn | [link](https://www.youtube.com/channel/UCIIkOzwZIdLfHa4Q13dQrkw/videos) | JhBv-6J2og8, jrd45Rbd6EA |
| Prothom Alo ✓ | news | BD | standard | 5.99 M | 1080p | hi | [link](https://www.youtube.com/channel/UCeG7m5-AJ4I0H4EIleIty4Q/videos) | mojX5qu36OY |
| The Daily Star ✓ | news | BD | standard | 2.73 M | 1080p | bn | [link](https://www.youtube.com/channel/UCKSJ5rTXq-pYgAmAOhVS_zQ/videos) | OfROGwmrDbU, Y0OxCffKKeQ |
| DBC NEWS ✓ | news | BD | standard | 8.46 M | 1080p | None, en | [link](https://www.youtube.com/channel/UCUvXoiDEKI8VZJrr58g4VAw/videos) | DN2VjqMIJFs, fM9E3buJljc |
| Channel i News ✓ | news | BD | standard | 6.46 M | 1080p | bn | [link](https://www.youtube.com/channel/UC8NcXMG3A3f2aFQyGTpSNww/videos) | hzFbB00oyeo, gpan3rV8zzk |
| ABP ANANDA ✓ | news | WB | standard | 14.50 M | 1080p | bn | [link](https://www.youtube.com/channel/UCv3rFzn-GHGtqzXiaq3sWNg/videos) | ciIOBCbyfVs, eM1n09YuBtA |
| Zee 24 Ghanta ✓ | news | WB | standard | 7.47 M | 1080p | bn | [link](https://www.youtube.com/channel/UCdF5Q5QVbYstYrTfpgUl0ZA/videos) | QycSed6wxdg, cBsQuSZuE20 |
| TV9 Bangla ✓ | news | WB | standard | 6.22 M | 1080p | bn | [link](https://www.youtube.com/channel/UCHCR4UFsGwd_VcDa0-a4haw/videos) | DQvfOF6n8Zs, IxY93I4wJxU |
| News18 Bangla ✓ | news | WB | standard | 13.20 M | 1080p | bn | [link](https://www.youtube.com/channel/UCbf0XHULBkTfv2hBjaaDw9Q/videos) | 0UKfqjsD2RE, uVFF8mVsXzg |
| NTV News ✓ | news/talk | BD | standard | 9.06 M | 1080p | bn | [link](https://www.youtube.com/channel/UCUDQdVsKssximyFwg4IxnOQ/videos) | H-cE66560ck, vDAGFjtQUrQ |
| SOMOY TV Bulletin ✓ | news/talk | BD | standard | 10.00 M | 1080p | bn | [link](https://www.youtube.com/channel/UCNUFterLJ9vpFZZ0try7sLA/videos) | S3tsm_faalA, KsnT1jid548 |
| Kaler Kantho ✓ | talk | BD | standard | 1.61 M | 1080p | bn | [link](https://www.youtube.com/channel/UCsBqX8QguRdkawdlNbIL1dw/videos) | aNKOJObCMcw |
| Deepto News ✓ | talk | BD | standard | 1.27 M | 1080p | bn | [link](https://www.youtube.com/channel/UC8uzDYcztPf-fICdGTowAAA/videos) | AVdizUiwjk0, UA2f0xhgaKQ |
| The Talk Show ✓ | talk | BD | standard | 301 k | 1080p | en | [link](https://www.youtube.com/channel/UCMXaxzsQSvnwyIsC-bMgiCw/videos) | wtNHCiKlPC0, 1Xl-aFf9NWY |
| Podcast with Arijit Chakraborty (Bengali Version) ✓ | podcast | WB | standard | 906 k | 2160p | bn | [link](https://www.youtube.com/channel/UCCyP1bZGzPgFAOjTuvVPH0w/videos) | 5X7dTmeKXA8, Mil9pxmiVTU |
| Ss Sunny Convos (Bengali Podcast) | podcast | ? | CC BY | 36 k | 2160p | bn | [link](https://www.youtube.com/channel/UCNXPePOUo1uB2ffsFWVMNhA/videos) | XxXRPPtVHLI, 6HDwg53-ZM0 |
| SameerScane ✓ | podcast | BD | standard | 123 k | 2160p | bn | [link](https://www.youtube.com/channel/UCW5yX7_V4lBgyhAELf--cxA/videos) | 1IPnKq1trN4, YpwN65fzwco |
| Time and Tide Podcast with Sourav | podcast | WB | standard | 346 k | 1080p | bn | [link](https://www.youtube.com/channel/UCKfp1FrO5l9T9TDPE4H5_ng/videos) | XjEMzHUA-Dc, vl2Zru50MQA |
| Nur Rahman Podcast | podcast | BD | standard | 107 k | 1080p | bn | [link](https://www.youtube.com/channel/UChiJqc4XY8bTmlj8gfL0KFQ/videos) | 5CTVRcJE9Lw |
| The Trinomial Podcast ✓ | podcast | BD | standard | 103 k | 1080p | bn | [link](https://www.youtube.com/channel/UCLU_ZSD373JBRqVNsqHVJFw/videos) | dD1oa-tCIjA |
| Esha Rushdi | podcast | BD | standard | 84 k | 1080p | bn, en-US | [link](https://www.youtube.com/channel/UCguZQBWNXQmldkl6lA-bAOg/videos) | nPzG_n2LPk4, NA6SsGagQIw |
| Songe Sangita ✓ | podcast | WB | standard | 202 k | 2160p | bn | [link](https://www.youtube.com/channel/UC4I4k41dSerMi6TOxfFZHZw/videos) | XjQZkcqMbLY, AIcMEZDh0XI |
| UNSCRIPTED talk with Santadip ✓ | podcast | WB | standard | 307 k | 1080p | bn | [link](https://www.youtube.com/channel/UCSCDSgmIc-uBIU6xf6SevDQ/videos) | o7q_gpEqN3g |
| D Talks | podcast | ? | standard | 116 k | 1080p | en | [link](https://www.youtube.com/channel/UCYGztcCmHTr8mRQuE1ogMAA/videos) | yyZxDmfMchM |
| 10 Minute School ✓ | education | BD | standard | 3.55 M | 1080p | bn | [link](https://www.youtube.com/channel/UCL89KKkLs0tZKld-iIS3NGw/videos) | G007RBO6yOo |
| Bangla Gurukul, GOLN | education | BD | standard | 103 k | 1080p | bn, hi | [link](https://www.youtube.com/channel/UCQAnpUMLyCSzTg60QlsQjZg/videos) | wxnO-QSYYeA, 0SNX8WnL_GU |
| Bishal Dar Class | education | ? | standard | 53 k | 1080p | bn | [link](https://www.youtube.com/channel/UCYPs4ObvwenWKu7RIpnj1cA/videos) | mEsAhVH8MlI, Wi62pwn8nlY |
| TEDx Talks ✓ | tedx | intl | standard | 44.70 M | 1080p | bn, en | [link](https://www.youtube.com/channel/UCsT0YIqwnpJCM-mx7-gSA4Q/videos) | aZku4Y4BKAM, vqJHF_maQ7U |
| পিস টিভি বাংলা - Peace TV Bangla | religious lecture | ? | CC BY | 444 k | 720p | None, hi | [link](https://www.youtube.com/channel/UCClsYw7y1BTTtTT8-EE_GPw/videos) | SiUEEAx5z48, -0LcJyr6Gos |
| Ahmadullah ✓ | religious lecture | BD | CC BY | 3.56 M | 1080p | bn | [link](https://www.youtube.com/channel/UCuxth2BimHUigZ344JhcFPw/videos) | I_gEpAXJIfA, byPFFnaMNUg |

The same list, ready for `bvsr.download expand`, is in `links/candidate_channels.txt`. Use
`--max-per-link` so each channel contributes only its newest N videos.

## 5. Balance targets

Existing Bengali datasets are male-heavy: LipBengal 92% male, and BenAV 107 male / 21 female
speakers. A better dataset should report and control:

| Dimension | Plan |
|---|---|
| Gender | target at least 35–40% female clips. News channels have many female anchors; deliberately add female-hosted podcasts and interviews. Annotate gender **per speaker by hand** (from face clusters), not from names or automatic gender classifiers, and report it like LRW-Persian's metadata. |
| Region / dialect | both Bangladesh and West Bengal channels; record the region per channel |
| Genre | report hours and clips per genre |
| Speakers | cap clips per speaker and per channel so a few anchors don't dominate |
| Splits | speaker- or channel-disjoint test set |

## 6. Screening step (proposed, not run)

Rank channels by how much usable footage they have, before bulk downloading:
1. For each candidate channel, take 3 recent videos.
2. Download only a 2-minute section of each at 480p (`yt-dlp --download-sections`).
3. Run `bvsr.faces`, and per video measure: share of frames with exactly one frontal face (|yaw| ≤ 30°, ≥100 px), shot cuts per minute, and speech share from VAD.
4. Keep channels where most frames pass, and note the host's gender by looking at the faces.

This means about 110 short downloads, so it needs a go-ahead before running.

## 7. Choosing the words

| Dataset | Vocabulary rule |
|---|---|
| LRW | the 500 most frequent words **5–10 characters long**, so the word fits the 1.16 s clip; each word ≥800 training and ≥40 val/test occurrences |
| LRW-1000 | 1,000 classes, *naturally distributed* (unbalanced, real frequencies) |
| LRW-AR | 100 words |
| LRW-Persian | top 2,500 words per programme, intersected across channels, pruned by hand to 743 |

**Proposal for Bengali:**
1. Run the pilot and bulk collection, then use `bvsr.cut stats` to count every word's usable clips, **distinct speakers and distinct channels**.
2. **Length rule by duration, not characters.** Bengali characters (conjuncts, vowel signs) don't map to letters the way English does. Keep words whose median aligned duration fits the clip (about 0.25–0.9 s) and that have at least 3 grapheme clusters, so very short function words drop out, as LRW's 5-character minimum does.
3. **Require spread:** a word must occur with many speakers and several channels, e.g. at least 20 speakers and 3 channels, so the model learns the word rather than one person.
4. **Exclude** numbers, proper names (people and places that are only frequent because of the news cycle), and possibly English loanwords; to decide, they could be kept as a separate flagged subset.
5. **Homophenes:** group candidate words by viseme sequence (thesis Table 4.4 mapping) and report the visually identical groups, e.g. কলম / গরম.
6. **Size:** LRW-style balanced, 500 words × at least 200 clips, as the main benchmark; optionally a naturally distributed release of everything, as LRW-1000 does. The final count should follow the frequency curve from the pilot.
7. Bengali speakers review the final word list.
