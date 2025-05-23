/* sum02.f -- translated by f2c (version 19980913).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Table of constant values */

static integer c__3 = 3;
static integer c__2 = 2;

/* $Procedure   SUM02 ( DSKBRIEF, summarize type 2 segment ) */
/* Subroutine */ int sum02_(integer *handle, integer *dladsc, integer *nsig)
{
    /* Initialized data */

    static char coord[1*3] = "X" "Y" "Z";

    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_rnge(char *, integer, char *, integer);

    /* Local variables */
    integer ncgr, i__;
    extern /* Subroutine */ int dskb02_(integer *, integer *, integer *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    integer *, integer *, integer *, integer *, integer *);
    char table[132*3];
    extern /* Subroutine */ int chkin_(char *, ftnlen), repmc_(char *, char *,
	     char *, char *, ftnlen, ftnlen, ftnlen, ftnlen), repmf_(char *, 
	    char *, doublereal *, integer *, char *, char *, ftnlen, ftnlen, 
	    ftnlen, ftnlen), repmi_(char *, char *, integer *, char *, ftnlen,
	     ftnlen, ftnlen);
    integer start1, cgscal, np;
    char labels[132*3];
    integer nv;
    extern /* Subroutine */ int cortab_(integer *, char *, integer *, integer 
	    *, integer *, doublereal *, integer *, char *, ftnlen, ftnlen);
    extern logical return_(void);
    char outlin[132];
    doublereal voxori[3], voxsiz, vtxbds[6]	/* was [2][3] */;
    integer cgrext[3], nvxtot, starts[2], voxnpl, voxnpt, vtxnpl, vgrext[3];
    extern /* Subroutine */ int tostdo_(char *, ftnlen), setmsg_(char *, 
	    ftnlen), errint_(char *, integer *, ftnlen), sigerr_(char *, 
	    ftnlen), chkout_(char *, ftnlen);

/* $ Abstract */

/*     Display type 2-specific summary of contents of a SPICE */
/*     Digital Shape Kernel (DSK). */

/* $ Disclaimer */

/*     THIS SOFTWARE AND ANY RELATED MATERIALS WERE CREATED BY THE */
/*     CALIFORNIA INSTITUTE OF TECHNOLOGY (CALTECH) UNDER A U.S. */
/*     GOVERNMENT CONTRACT WITH THE NATIONAL AERONAUTICS AND SPACE */
/*     ADMINISTRATION (NASA). THE SOFTWARE IS TECHNOLOGY AND SOFTWARE */
/*     PUBLICLY AVAILABLE UNDER U.S. EXPORT LAWS AND IS PROVIDED "AS-IS" */
/*     TO THE RECIPIENT WITHOUT WARRANTY OF ANY KIND, INCLUDING ANY */
/*     WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A */
/*     PARTICULAR USE OR PURPOSE (AS SET FORTH IN UNITED STATES UCC */
/*     SECTIONS 2312-2313) OR FOR ANY PURPOSE WHATSOEVER, FOR THE */
/*     SOFTWARE AND RELATED MATERIALS, HOWEVER USED. */

/*     IN NO EVENT SHALL CALTECH, ITS JET PROPULSION LABORATORY, OR NASA */
/*     BE LIABLE FOR ANY DAMAGES AND/OR COSTS, INCLUDING, BUT NOT */
/*     LIMITED TO, INCIDENTAL OR CONSEQUENTIAL DAMAGES OF ANY KIND, */
/*     INCLUDING ECONOMIC DAMAGE OR INJURY TO PROPERTY AND LOST PROFITS, */
/*     REGARDLESS OF WHETHER CALTECH, JPL, OR NASA BE ADVISED, HAVE */
/*     REASON TO KNOW, OR, IN FACT, SHALL KNOW OF THE POSSIBILITY. */

/*     RECIPIENT BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF */
/*     THE SOFTWARE AND ANY RELATED MATERIALS, AND AGREES TO INDEMNIFY */
/*     CALTECH AND NASA FOR ALL THIRD-PARTY CLAIMS RESULTING FROM THE */
/*     ACTIONS OF RECIPIENT IN THE USE OF THE SOFTWARE. */

/* $ Required_Reading */

/*     DSK */

/* $ Keywords */

/*     DSK */
/*     DSKBRIEF */

/* $ Declarations */

/*     Include file dla.inc */

/*     This include file declares parameters for DLA format */
/*     version zero. */

/*        Version 3.0.1 17-OCT-2016 (NJB) */

/*           Corrected comment: VERIDX is now described as a DAS */
/*           integer address rather than a d.p. address. */

/*        Version 3.0.0 20-JUN-2006 (NJB) */

/*           Changed name of parameter DSCSIZ to DLADSZ. */

/*        Version 2.0.0 09-FEB-2005 (NJB) */

/*           Changed descriptor layout to make backward pointer */
/*           first element.  Updated DLA format version code to 1. */

/*           Added parameters for format version and number of bytes per */
/*           DAS comment record. */

/*        Version 1.0.0 28-JAN-2004 (NJB) */


/*     DAS integer address of DLA version code. */


/*     Linked list parameters */

/*     Logical arrays (aka "segments") in a DAS linked array (DLA) file */
/*     are organized as a doubly linked list.  Each logical array may */
/*     actually consist of character, double precision, and integer */
/*     components.  A component of a given data type occupies a */
/*     contiguous range of DAS addresses of that type.  Any or all */
/*     array components may be empty. */

/*     The segment descriptors in a SPICE DLA (DAS linked array) file */
/*     are connected by a doubly linked list.  Each node of the list is */
/*     represented by a pair of integers acting as forward and backward */
/*     pointers.  Each pointer pair occupies the first two integers of a */
/*     segment descriptor in DAS integer address space.  The DLA file */
/*     contains pointers to the first integers of both the first and */
/*     last segment descriptors. */

/*     At the DLA level of a file format implementation, there is */
/*     no knowledge of the data contents.  Hence segment descriptors */
/*     provide information only about file layout (in contrast with */
/*     the DAF system).  Metadata giving specifics of segment contents */
/*     are stored within the segments themselves in DLA-based file */
/*     formats. */


/*     Parameter declarations follow. */

/*     DAS integer addresses of first and last segment linked list */
/*     pointer pairs.  The contents of these pointers */
/*     are the DAS addresses of the first integers belonging */
/*     to the first and last link pairs, respectively. */

/*     The acronyms "LLB" and "LLE" denote "linked list begin" */
/*     and "linked list end" respectively. */


/*     Null pointer parameter. */


/*     Segment descriptor parameters */

/*     Each segment descriptor occupies a contiguous */
/*     range of DAS integer addresses. */

/*        The segment descriptor layout is: */

/*           +---------------+ */
/*           | BACKWARD PTR  | Linked list backward pointer */
/*           +---------------+ */
/*           | FORWARD PTR   | Linked list forward pointer */
/*           +---------------+ */
/*           | BASE INT ADDR | Base DAS integer address */
/*           +---------------+ */
/*           | INT COMP SIZE | Size of integer segment component */
/*           +---------------+ */
/*           | BASE DP ADDR  | Base DAS d.p. address */
/*           +---------------+ */
/*           | DP COMP SIZE  | Size of d.p. segment component */
/*           +---------------+ */
/*           | BASE CHR ADDR | Base DAS character address */
/*           +---------------+ */
/*           | CHR COMP SIZE | Size of character segment component */
/*           +---------------+ */

/*     Parameters defining offsets for segment descriptor elements */
/*     follow. */


/*     Descriptor size: */


/*     Other DLA parameters: */


/*     DLA format version.  (This number is expected to occur very */
/*     rarely at integer address VERIDX in uninitialized DLA files.) */


/*     Characters per DAS comment record. */


/*     End of include file dla.inc */


/*     Include file dskdsc.inc */

/*     This include file declares parameters for DSK segment descriptors. */

/* -       SPICELIB Version 1.0.0 08-FEB-2017 (NJB) */

/*           Updated version info. */

/*           22-JAN-2016 (NJB) */

/*              Added parameter for data class 2. Changed name of data */
/*              class 1 parameter. Corrected data class descriptions. */

/*           13-MAY-2010 (NJB) */

/*              Descriptor now contains two ID codes, one for the */
/*              surface, one for the associated ephemeris object. This */
/*              supports association of multiple surfaces with one */
/*              ephemeris object without creating file management */
/*              issues. */

/*              Room was added for coordinate system definition */
/*              parameters. */

/*               Flag arrays and model ID/component entries were deleted. */

/*            11-SEP-2008 (NJB) */


/*     DSK segment descriptors are implemented as an array of d.p. */
/*     numbers.  Note that each integer descriptor datum occupies one */
/*     d.p. value. */




/*     Segment descriptor parameters */

/*     Each segment descriptor occupies a contiguous */
/*     range of DAS d.p. addresses. */

/*        The DSK segment descriptor layout is: */

/*           +---------------------+ */
/*           | Surface ID code     | */
/*           +---------------------+ */
/*           | Center ID code      | */
/*           +---------------------+ */
/*           | Data class code     | */
/*           +---------------------+ */
/*           | Data type           | */
/*           +---------------------+ */
/*           | Ref frame code      | */
/*           +---------------------+ */
/*           | Coord sys code      | */
/*           +---------------------+ */
/*           | Coord sys parameters|  {10 elements} */
/*           +---------------------+ */
/*           | Min coord 1         | */
/*           +---------------------+ */
/*           | Max coord 1         | */
/*           +---------------------+ */
/*           | Min coord 2         | */
/*           +---------------------+ */
/*           | Max coord 2         | */
/*           +---------------------+ */
/*           | Min coord 3         | */
/*           +---------------------+ */
/*           | Max coord 3         | */
/*           +---------------------+ */
/*           | Start time          | */
/*           +---------------------+ */
/*           | Stop time           | */
/*           +---------------------+ */

/*     Parameters defining offsets for segment descriptor elements */
/*     follow. */


/*     Surface ID code: */


/*     Central ephemeris object NAIF ID: */


/*     Data class: */

/*     The "data class" is a code indicating the category of */
/*     data contained in the segment. */


/*     Data type: */


/*     Frame ID: */


/*     Coordinate system code: */


/*     Coordinate system parameter start index: */


/*     Number of coordinate system parameters: */


/*     Ranges for coordinate bounds: */


/*     Coverage time bounds: */


/*     Descriptor size (24): */


/*     Data class values: */

/*        Class 1 indicates a surface that can be represented as a */
/*                single-valued function of its domain coordinates. */

/*                An example is a surface defined by a function that */
/*                maps each planetodetic longitude and latitude pair to */
/*                a unique altitude. */


/*        Class 2 indicates a general surface. Surfaces that */
/*                have multiple points for a given pair of domain */
/*                coordinates---for example, multiple radii for a given */
/*                latitude and longitude---belong to class 2. */



/*     Coordinate system values: */

/*        The coordinate system code indicates the system to which the */
/*        tangential coordinate bounds belong. */

/*        Code 1 refers to the planetocentric latitudinal system. */

/*        In this system, the first tangential coordinate is longitude */
/*        and the second tangential coordinate is latitude. The third */
/*        coordinate is radius. */



/*        Code 2 refers to the cylindrical system. */

/*        In this system, the first tangential coordinate is radius and */
/*        the second tangential coordinate is longitude. The third, */
/*        orthogonal coordinate is Z. */



/*        Code 3 refers to the rectangular system. */

/*        In this system, the first tangential coordinate is X and */
/*        the second tangential coordinate is Y. The third, */
/*        orthogonal coordinate is Z. */



/*        Code 4 refers to the planetodetic/geodetic system. */

/*        In this system, the first tangential coordinate is longitude */
/*        and the second tangential coordinate is planetodetic */
/*        latitude. The third, orthogonal coordinate is altitude. */



/*     End of include file dskdsc.inc */


/*     Include file dsk02.inc */

/*     This include file declares parameters for DSK data type 2 */
/*     (plate model). */

/* -       SPICELIB Version 1.0.0 08-FEB-2017 (NJB) */

/*          Updated version info. */

/*           22-JAN-2016 (NJB) */

/*              Now includes spatial index parameters. */

/*           26-MAR-2015 (NJB) */

/*              Updated to increase MAXVRT to 16000002. MAXNPV */
/*              has been changed to (3/2)*MAXPLT. Set MAXVOX */
/*              to 100000000. */

/*           13-MAY-2010 (NJB) */

/*              Updated to reflect new no-record design. */

/*           04-MAY-2010 (NJB) */

/*              Updated for new type 2 segment design. Now uses */
/*              a local parameter to represent DSK descriptor */
/*              size (NB). */

/*           13-SEP-2008 (NJB) */

/*              Updated to remove albedo information. */
/*              Updated to use parameter for DSK descriptor size. */

/*           27-DEC-2006 (NJB) */

/*              Updated to remove minimum and maximum radius information */
/*              from segment layout.  These bounds are now included */
/*              in the segment descriptor. */

/*           26-OCT-2006 (NJB) */

/*              Updated to remove normal, center, longest side, albedo, */
/*              and area keyword parameters. */

/*           04-AUG-2006 (NJB) */

/*              Updated to support coarse voxel grid.  Area data */
/*              have now been removed. */

/*           10-JUL-2006 (NJB) */


/*     Each type 2 DSK segment has integer, d.p., and character */
/*     components.  The segment layout in DAS address space is as */
/*     follows: */


/*        Integer layout: */

/*           +-----------------+ */
/*           | NV              |  (# of vertices) */
/*           +-----------------+ */
/*           | NP              |  (# of plates ) */
/*           +-----------------+ */
/*           | NVXTOT          |  (total number of voxels) */
/*           +-----------------+ */
/*           | VGREXT          |  (voxel grid extents, 3 integers) */
/*           +-----------------+ */
/*           | CGRSCL          |  (coarse voxel grid scale, 1 integer) */
/*           +-----------------+ */
/*           | VOXNPT          |  (size of voxel-plate pointer list) */
/*           +-----------------+ */
/*           | VOXNPL          |  (size of voxel-plate list) */
/*           +-----------------+ */
/*           | VTXNPL          |  (size of vertex-plate list) */
/*           +-----------------+ */
/*           | PLATES          |  (NP 3-tuples of vertex IDs) */
/*           +-----------------+ */
/*           | VOXPTR          |  (voxel-plate pointer array) */
/*           +-----------------+ */
/*           | VOXPLT          |  (voxel-plate list) */
/*           +-----------------+ */
/*           | VTXPTR          |  (vertex-plate pointer array) */
/*           +-----------------+ */
/*           | VTXPLT          |  (vertex-plate list) */
/*           +-----------------+ */
/*           | CGRPTR          |  (coarse grid occupancy pointers) */
/*           +-----------------+ */



/*        D.p. layout: */

/*           +-----------------+ */
/*           | DSK descriptor  |  DSKDSZ elements */
/*           +-----------------+ */
/*           | Vertex bounds   |  6 values (min/max for each component) */
/*           +-----------------+ */
/*           | Voxel origin    |  3 elements */
/*           +-----------------+ */
/*           | Voxel size      |  1 element */
/*           +-----------------+ */
/*           | Vertices        |  3*NV elements */
/*           +-----------------+ */


/*     This local parameter MUST be kept consistent with */
/*     the parameter DSKDSZ which is declared in dskdsc.inc. */


/*     Integer item keyword parameters used by fetch routines: */


/*     Double precision item keyword parameters used by fetch routines: */


/*     The parameters below formerly were declared in pltmax.inc. */

/*     Limits on plate model capacity: */

/*     The maximum number of bodies, vertices and */
/*     plates in a plate model or collective thereof are */
/*     provided here. */

/*     These values can be used to dimension arrays, or to */
/*     use as limit checks. */

/*     The value of MAXPLT is determined from MAXVRT via */
/*     Euler's Formula for simple polyhedra having triangular */
/*     faces. */

/*     MAXVRT is the maximum number of vertices the triangular */
/*            plate model software will support. */


/*     MAXPLT is the maximum number of plates that the triangular */
/*            plate model software will support. */


/*     MAXNPV is the maximum allowed number of vertices, not taking into */
/*     account shared vertices. */

/*     Note that this value is not sufficient to create a vertex-plate */
/*     mapping for a model of maximum plate count. */


/*     MAXVOX is the maximum number of voxels. */


/*     MAXCGR is the maximum size of the coarse voxel grid. */


/*     MAXEDG is the maximum allowed number of vertex or plate */
/*     neighbors a vertex may have. */

/*     DSK type 2 spatial index parameters */
/*     =================================== */

/*        DSK type 2 spatial index integer component */
/*        ------------------------------------------ */

/*           +-----------------+ */
/*           | VGREXT          |  (voxel grid extents, 3 integers) */
/*           +-----------------+ */
/*           | CGRSCL          |  (coarse voxel grid scale, 1 integer) */
/*           +-----------------+ */
/*           | VOXNPT          |  (size of voxel-plate pointer list) */
/*           +-----------------+ */
/*           | VOXNPL          |  (size of voxel-plate list) */
/*           +-----------------+ */
/*           | VTXNPL          |  (size of vertex-plate list) */
/*           +-----------------+ */
/*           | CGRPTR          |  (coarse grid occupancy pointers) */
/*           +-----------------+ */
/*           | VOXPTR          |  (voxel-plate pointer array) */
/*           +-----------------+ */
/*           | VOXPLT          |  (voxel-plate list) */
/*           +-----------------+ */
/*           | VTXPTR          |  (vertex-plate pointer array) */
/*           +-----------------+ */
/*           | VTXPLT          |  (vertex-plate list) */
/*           +-----------------+ */


/*        Index parameters */


/*     Grid extent: */


/*     Coarse grid scale: */


/*     Voxel pointer count: */


/*     Voxel-plate list count: */


/*     Vertex-plate list count: */


/*     Coarse grid pointers: */


/*     Size of fixed-size portion of integer component: */


/*        DSK type 2 spatial index double precision component */
/*        --------------------------------------------------- */

/*           +-----------------+ */
/*           | Vertex bounds   |  6 values (min/max for each component) */
/*           +-----------------+ */
/*           | Voxel origin    |  3 elements */
/*           +-----------------+ */
/*           | Voxel size      |  1 element */
/*           +-----------------+ */



/*        Index parameters */

/*     Vertex bounds: */


/*     Voxel grid origin: */


/*     Voxel size: */


/*     Size of fixed-size portion of double precision component: */


/*     The limits below are used to define a suggested maximum */
/*     size for the integer component of the spatial index. */


/*     Maximum number of entries in voxel-plate pointer array: */


/*     Maximum cell size: */


/*     Maximum number of entries in voxel-plate list: */


/*     Spatial index integer component size: */


/*     End of include file dsk02.inc */

/* $ Brief_I/O */

/*     Variable  I/O  Description */
/*     --------  ---  -------------------------------------------------- */
/*     HANDLE     I   Handle of DSK file. */
/*     DLADSC     I   DLA descriptor of segment. */
/*     NSIG       I   Number of significant digits in floating point */
/*                    output. */

/* $ Detailed_Input */

/*     HANDLE     is the handle of a DSK file containing a segment */
/*                to be summarized. */

/*     DLADSC     is the DLA descriptor of a segment to be summarized. */

/*     NSIG       is the number of significant digits in floating point */
/*                numeric output. The range of NSIG is 6:17. */

/* $ Detailed_Output */

/*     None. */

/* $ Parameters */

/*     None. */

/* $ Exceptions */

/*     1)  If an invalid coarse voxel scale is encountered, this routine */
/*         signals the error SPICE(VALUEOUTOFRANGE). */

/* $ Files */

/*     See the input HANDLE. */

/* $ Particulars */

/*     This routine displays detailed summary information for a */
/*     specified type 2 DSK segment. The display is written to */
/*     standard output. */

/* $ Examples */

/*     None. */

/* $ Restrictions */

/*     None. */

/* $ Literature_References */

/*     None. */

/* $ Author_and_Institution */

/*     N.J. Bachman   (JPL) */

/* $ Version */

/* -    DSKBRIEF Version 1.0.0, 05-OCT-2016 (NJB) */

/* -& */
/* $ Index_Entries */

/*     summarize type 2 dsk segment */

/* -& */

/*     SPICELIB functions */


/*     Local parameters */


/*     Local variables */


/*     Saved variables */

/*     Note:  since this is a main program, SAVE statements are not */
/*     required.  They are used in case some of this code is later */
/*     ported to subroutines. */


/*     Initial values */

    if (return_()) {
	return 0;
    }
    chkin_("SUM02", (ftnlen)5);

/*     Display type 2 parameters. */

    tostdo_(" ", (ftnlen)1);
    tostdo_("    Type 2 parameters", (ftnlen)21);
    tostdo_("    -----------------", (ftnlen)21);
    dskb02_(handle, dladsc, &nv, &np, &nvxtot, vtxbds, &voxsiz, voxori, 
	    vgrext, &cgscal, &vtxnpl, &voxnpt, &voxnpl);

/*     Show vertex and plate counts. */

    s_copy(outlin, "      Number of vertices:                 #", (ftnlen)132,
	     (ftnlen)43);
    repmi_(outlin, "#", &nv, outlin, (ftnlen)132, (ftnlen)1, (ftnlen)132);
    tostdo_(outlin, (ftnlen)132);
    s_copy(outlin, "      Number of plates:                   #", (ftnlen)132,
	     (ftnlen)43);
    repmi_(outlin, "#", &np, outlin, (ftnlen)132, (ftnlen)1, (ftnlen)132);
    tostdo_(outlin, (ftnlen)132);

/*     Show voxel size. */

    s_copy(outlin, "      Voxel edge length (km):             #", (ftnlen)132,
	     (ftnlen)43);
    repmf_(outlin, "#", &voxsiz, nsig, "E", outlin, (ftnlen)132, (ftnlen)1, (
	    ftnlen)1, (ftnlen)132);
    tostdo_(outlin, (ftnlen)132);

/*     Show voxel grid count. */

    s_copy(outlin, "      Number of voxels:                   #", (ftnlen)132,
	     (ftnlen)43);
    repmi_(outlin, "#", &nvxtot, outlin, (ftnlen)132, (ftnlen)1, (ftnlen)132);
    tostdo_(outlin, (ftnlen)132);

/*     Show coarse voxel grid count. */

    if (cgscal < 1) {
	setmsg_("Coarse voxel grid scale is #; this scale should be an integ"
		"er > 0", (ftnlen)65);
	errint_("#", &cgscal, (ftnlen)1);
	sigerr_("SPICE(VALUEOUTOFRANGE)", (ftnlen)22);
    }
    ncgr = 1;
    for (i__ = 1; i__ <= 3; ++i__) {
	cgrext[(i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 : s_rnge("cgrext", 
		i__1, "sum02_", (ftnlen)241)] = vgrext[(i__2 = i__ - 1) < 3 &&
		 0 <= i__2 ? i__2 : s_rnge("vgrext", i__2, "sum02_", (ftnlen)
		241)] / cgscal;
	ncgr *= cgrext[(i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 : s_rnge(
		"cgrext", i__1, "sum02_", (ftnlen)242)];
    }
    s_copy(outlin, "      Number of coarse voxels:            #", (ftnlen)132,
	     (ftnlen)43);
    repmi_(outlin, "#", &ncgr, outlin, (ftnlen)132, (ftnlen)1, (ftnlen)132);
    tostdo_(outlin, (ftnlen)132);

/*     Show voxel grid extents. */

    s_copy(outlin, "      Voxel grid X, Y, Z extents:         # # #", (ftnlen)
	    132, (ftnlen)47);
    for (i__ = 1; i__ <= 3; ++i__) {
	repmi_(outlin, "#", &vgrext[(i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 :
		 s_rnge("vgrext", i__1, "sum02_", (ftnlen)255)], outlin, (
		ftnlen)132, (ftnlen)1, (ftnlen)132);
    }
    tostdo_(outlin, (ftnlen)132);

/*     Show coarse voxel grid extents. */

    s_copy(outlin, "      Coarse voxel grid X, Y, Z extents:  # # #", (ftnlen)
	    132, (ftnlen)47);
    for (i__ = 1; i__ <= 3; ++i__) {
	repmi_(outlin, "#", &cgrext[(i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 :
		 s_rnge("cgrext", i__1, "sum02_", (ftnlen)267)], outlin, (
		ftnlen)132, (ftnlen)1, (ftnlen)132);
    }
    tostdo_(outlin, (ftnlen)132);

/*     Show vertex bounds. */

    for (i__ = 1; i__ <= 3; ++i__) {
	s_copy(labels + ((i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 : s_rnge(
		"labels", i__1, "sum02_", (ftnlen)278)) * 132, "      Min, m"
		"ax vertex # value (km):", (ftnlen)132, (ftnlen)35);
	repmc_(labels + ((i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 : s_rnge(
		"labels", i__1, "sum02_", (ftnlen)279)) * 132, "#", coord + ((
		i__2 = i__ - 1) < 3 && 0 <= i__2 ? i__2 : s_rnge("coord", 
		i__2, "sum02_", (ftnlen)279)), labels + ((i__3 = i__ - 1) < 3 
		&& 0 <= i__3 ? i__3 : s_rnge("labels", i__3, "sum02_", (
		ftnlen)279)) * 132, (ftnlen)132, (ftnlen)1, (ftnlen)1, (
		ftnlen)132);
    }
    start1 = 43;
    cortab_(&c__3, labels, &start1, nsig, &c__2, vtxbds, starts, table, (
	    ftnlen)132, (ftnlen)132);
    for (i__ = 1; i__ <= 3; ++i__) {
	tostdo_(table + ((i__1 = i__ - 1) < 3 && 0 <= i__1 ? i__1 : s_rnge(
		"table", i__1, "sum02_", (ftnlen)289)) * 132, (ftnlen)132);
    }
    chkout_("SUM02", (ftnlen)5);
    return 0;
} /* sum02_ */

