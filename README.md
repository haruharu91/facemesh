# facemesh
Use open source face recognition to detect doppelgängers.

If you have a doppelgänger problem or an imposter issue, you should attempt to use fingerprints and DNA. Fingerprints already biologically identify distinct individual human beings, especially if all 10 fingerprints are taken. Face meshes are used to correlate face structure biometrics with sets of 10 fingerprints. DNA testing is ultimately also a good test for identity but sets of 10 fingerprints are adequate to definitely separate out individuals. Fingerprint minutiae from full fingerprint sets can even distinguish identical twins because of small differences in development. Consult with your attorney if you're having that sort of problem and consider hiring a private investigator.

One use case for this is detecting alternate actors in films. Actors often use scene doubles and not just stunt doubles: having 2 actors who are closer doppelgänger match and who are both highly-skilled actors means directors and production companies have a sort of insurance for the completion of the film product. This is even reflected in contracts for major motion pictures. An actor who has a few alternates can have more simultaneous projects and the project is less affected by illness or injury. A famous example is how Tom Hanks's brothers helped him finish Forrest Gump and did some entire segments of the films like the running scenes. Martin Sheen finished Apocalypse Now! after Harvey Keitel's heart attack but that involved reshoots. After Brandon Lee died on the set of The Crow, Chad Stahelski (future director of John Wick and Keanu Reeves's double in The Matrix) finished the part, but CGI was used to superimpose Brandon's face, so movies are unreliable information for face meshes. 

Once again ask your attorney: but face mesh distance can also detect political body doubles. Politicians can use a security strategy where they have a body double who draws out attacks intended for the politician. The underlying mechanism is the same as an actor's double. You should worry more if the real politician hasn't shown back up.

This looks for images uploaded into the /img folder, so from /bin you can run: 

$ python3 facemesh.py feigl1.jpg feigl2.jpg 

to receive an example output.

I've uploaded photos of the doppelgänger baseball players both named Brady FEIGL, who had the same Tommy John surgeon but aren't from the same family. One was born in 1990 and one was born in 1995 in different jurisdictions and they didn't know each other, but look similar, style themselves similarly, and both became baseball players. 

/img also contains an example output.

This uses FairFace because, although AI-assisted ethnicity inference is always sketchy, and seems to be especially biased against Japanese people, FairFace is supposed to be trained for less prejudice. Projects like Facebook's DeepFace are Eurocentric and lead to false negatives on pictures of women and people of colour. 

So that part might get removed if it's just too problematic, but these APIs do in fact attempt to infer race, age, and mood -- all of them very inaccurately.

The part that's more scientific is the face mesh distance: face mesh distance is actually used to find supsects using surveillance cameras. 

So metrics like the euclidean distances between the vector spaces of the 2 face meshes can in fact detect different people. 

The photos that are the same Brady Feigl have a small euclidean distance, like if you use feigl1.jpg and feigl3.jpg or feigl2.jpg and feigl4.jpg, but comparing odd and even results in a greater euclidean distance because they are biologically distinct and just an example from baseball of doppelgängers. 
