// Loaded on first open so the page does something before anything is typed.
// Small, and it works: the first thing anyone sees should be a clean result,
// not a wall of errors in code they did not write.
export const SAMPLE = `#include <stdio.h>

int main(void) {
    int total = 0;

    // Add up the numbers 1 to 5.
    for (int i = 1; i <= 5; i++) {
        total += i;
    }

    printf("The sum is %d\\n", total);
    return 0;
}
`;
